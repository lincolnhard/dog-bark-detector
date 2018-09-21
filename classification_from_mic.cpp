#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include <portaudio.h>

#include "window.h"
#include "common.h"
#include "spectrum.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "run_darknet.h"

static void
get_colour_map_value (float value, double spec_floor_db, unsigned char colour [3])
{	static unsigned char map [][3] =
    {	/* These values were originally calculated for a dynamic range of 180dB. */
        {	255,	255,	255	},	/* -0dB */
        {	240,	254,	216	},	/* -10dB */
        {	242,	251,	185	},	/* -20dB */
        {	253,	245,	143	},	/* -30dB */
        {	253,	200,	102	},	/* -40dB */
        {	252,	144,	66	},	/* -50dB */
        {	252,	75,		32	},	/* -60dB */
        {	237,	28,		41	},	/* -70dB */
        {	214,	3,		64	},	/* -80dB */
        {	183,	3,		101	},	/* -90dB */
        {	157,	3,		122	},	/* -100dB */
        {	122,	3,		126	},	/* -110dB */
        {	80,		2,		110	},	/* -120dB */
        {	45,		2,		89	},	/* -130dB */
        {	19,		2,		70	},	/* -140dB */
        {	1,		3,		53	},	/* -150dB */
        {	1,		3,		37	},	/* -160dB */
        {	1,		2,		19	},	/* -170dB */
        {	0,		0,		0	},	/* -180dB */
    } ;

    float rem ;
    int indx ;

    if (value >= 0.0)
    {	colour [0] = colour [1] = colour [2] = 255 ;
        return ;
        } ;

    value = fabs (value * (-180.0 / spec_floor_db) * 0.1) ;

    indx = lrintf (floor (value)) ;

    if (indx < 0)
    {	printf ("\nError : colour map array index is %d\n\n", indx) ;
        exit (1) ;
        } ;

    if (indx >= ARRAY_LEN (map) - 1)
    {	colour [0] = colour [1] = colour [2] = 0 ;
        return ;
        } ;

    rem = fmod (value, 1.0) ;

    colour [0] = lrintf ((1.0 - rem) * map [indx][0] + rem * map [indx + 1][0]) ;
    colour [1] = lrintf ((1.0 - rem) * map [indx][1] + rem * map [indx + 1][1]) ;
    colour [2] = lrintf ((1.0 - rem) * map [indx][2] + rem * map [indx + 1][2]) ;

    return ;
}

typedef struct
{	double value [40] ;  /* 35 or more */
    double distance [40] ;
    int decimal_places_to_print ;
} TICKS ;

#define TARGET_DIVISIONS 3

#define NO_NUMBER (M_PI)		/* They're unlikely to hit that! */

/* Is this entry in "ticks" one of the numberless ticks? */
#define JUST_A_TICK(ticks, k)	(ticks.value [k] == NO_NUMBER)

#define DELTA (1e-10)

static int	/* Forward declaration */
calculate_log_ticks (double min, double max, double distance, TICKS * ticks) ;

static int
calculate_ticks (double min, double max, double distance, int log_scale, TICKS * ticks)
{
    double step ;	/* Put numbered ticks at multiples of this */
    double range = max - min ;
    int k ;
    double value ;	/* Temporary */

    if (log_scale == 1)
        return calculate_log_ticks (min, max, distance, ticks) ;

    step = pow (10.0, floor (log10 (max))) ;
    do
    {	if (range / (step * 5) >= TARGET_DIVISIONS)
        {	step *= 5 ;
            break ;
            } ;
        if (range / (step * 2) >= TARGET_DIVISIONS)
        {	step *= 2 ;
            break ;
            } ;
        if (range / step >= TARGET_DIVISIONS)
            break ;
        step /= 10 ;
    } while (1) ;	/* This is an odd loop! */

    /* Ensure that the least significant digit that changes gets printed, */
    ticks->decimal_places_to_print = lrint (-floor (log10 (step))) ;
    if (ticks->decimal_places_to_print < 0)
        ticks->decimal_places_to_print = 0 ;

    /* Now go from the first multiple of step that's >= min to
     * the last one that's <= max. */
    k = 0 ;
    value = ceil (min / step) * step ;

#define add_tick(val, just_a_tick) do \
    {	if (val >= min - DELTA && val < max + DELTA) \
        {	ticks->value [k] = just_a_tick ? NO_NUMBER : val ; \
            ticks->distance [k] = distance * \
                (log_scale == 2 \
                    ? /*log*/ (log (val) - log (min)) / (log (max) - log (min)) \
                    : /*lin*/ (val - min) / range) ; \
            k++ ; \
            } ; \
        } while (0)

    /* Add the half-way tick before the first number if it's in range */
    add_tick (value - step / 2, true) ;

    while (value <= max + DELTA)
    { 	/* Add a tick next to each printed number */
        add_tick (value, false) ;

        /* and at the half-way tick after the number if it's in range */
        add_tick (value + step / 2, true) ;

        value += step ;
        } ;

    return k ;
} /* calculate_ticks */

static int
add_log_ticks (double min, double max, double distance, TICKS * ticks,
                int k, double start_value, bool include_number)
{	double value ;

    for (value = start_value ; value <= max + DELTA ; value *= 10.0)
    {	if (value < min - DELTA) continue ;
        ticks->value [k] = include_number ? value : NO_NUMBER ;
        ticks->distance [k] = distance * (log (value) - log (min)) / (log (max) - log (min)) ;
        k++ ;
        } ;
    return k ;
} /* add_log_ticks */

static int
calculate_log_ticks (double min, double max, double distance, TICKS * ticks)
{	int k = 0 ;	/* Number of ticks we have placed in "ticks" array */
    double underpinning ; 	/* Largest power of ten that is <= min */
    if (max / min < 10.0)
        return calculate_ticks (min, max, distance, 2, ticks) ;

    /* If the range is greater than 1 to 1000000, it will generate more than
    ** 19 ticks.  Better to fail explicitly than to overflow.
    */
    if (max / min > 1000000)
    {	printf ("Error: Frequency range is too great for logarithmic scale.\n") ;
        exit (1) ;
        } ;

    /* First hack: label the powers of ten. */

    /* Find largest power of ten that is <= minimum value */
    underpinning = pow (10.0, floor (log10 (min))) ;

    /* Go powering up by 10 from there, numbering as we go. */
    k = add_log_ticks (min, max, distance, ticks, k, underpinning, true) ;

    /* Do we have enough numbers? If so, add numberless ticks at 2 and 5 */
    if (k >= TARGET_DIVISIONS + 1) /* Number of labels is n.of divisions + 1 */
    {
        k = add_log_ticks (min, max, distance, ticks, k, underpinning * 2.0, false) ;
        k = add_log_ticks (min, max, distance, ticks, k, underpinning * 5.0, false) ;
        }
    else
    {	int i ;
        /* Not enough numbers: add numbered ticks at 2 and 5 and
         * unnumbered ticks at all the rest */
        for (i = 2 ; i <= 9 ; i++)
            k = add_log_ticks (min, max, distance, ticks, k,
                                underpinning * (1.0 * i), i == 2 || i == 5) ;
        } ;
    return k ;
} /* calculate_log_ticks */

static double
magindex_to_specindex (int speclen, int maglen, int magindex, double min_freq, double max_freq, int samplerate, bool log_freq)
{
    double freq ; /* The frequency that this output value represents */

    if (!log_freq)
        freq = min_freq + (max_freq - min_freq) * magindex / (maglen - 1) ;
    else
        freq = min_freq * pow (max_freq / min_freq, (double) magindex / (maglen - 1)) ;

    return (freq * speclen / (samplerate / 2)) ;
}

/* Map values from the spectrogram onto an array of magnitudes, the values
** for display. Reads spec[0..speclen], writes mag[0..maglen-1].
*/
static void
interp_spec (float * mag, int maglen, const double *spec, int speclen, const double min_freq, const double max_freq, int samplerate)
{
    int k ;
    for (k = 0 ; k < maglen ; k++)
    {	/* Average the pixels in the range it comes from */
        double current = magindex_to_specindex (speclen, maglen, k,
                        min_freq, max_freq, samplerate,
                        0.0) ;
        double next = magindex_to_specindex (speclen, maglen, k+1,
                        min_freq, max_freq, samplerate,
                        0.0) ;

        /* Range check: can happen if --max-freq > samplerate / 2 */
        if (current > speclen)
        {	mag [k] = 0.0 ;
            return ;
            } ;

        if (next > current + 1)
        {	/* The output indices are more sparse than the input indices
            ** so average the range of input indices that map to this output,
            ** making sure not to exceed the input array (0..speclen inclusive)
            */
            /* Take a proportional part of the first sample */
            double count = 1.0 - (current - floor (current)) ;
            double sum = spec [(int) current] * count ;

            while ((current += 1.0) < next && (int) current <= speclen)
            {	sum += spec [(int) current] ;
                count += 1.0 ;
                }
            /* and part of the last one */
            if ((int) next <= speclen)
            {	sum += spec [(int) next] * (next - floor (next)) ;
                count += next - floor (next) ;
                } ;

            mag [k] = sum / count ;
            }
        else
        /* The output indices are more densely packed than the input indices
        ** so interpolate between input values to generate more output values.
        */
        {	/* Take a weighted average of the nearest values */
            mag [k] = spec [(int) current] * (1.0 - (current - floor (current)))
                        + spec [(int) current + 1] * (current - floor (current)) ;
            } ;
        } ;

    return ;
}

static bool
is_2357 (int n)
{
    /* Just eliminate all factors os 2, 3, 5 and 7 and see if 1 remains */
    while (n % 2 == 0) n /= 2 ;
    while (n % 3 == 0) n /= 3 ;
    while (n % 5 == 0) n /= 5 ;
    while (n % 7 == 0) n /= 7 ;
    return (n == 1) ;
}

static bool
is_good_speclen (int n)
{
    /* It wants n, 11*n, 13*n but not (11*13*n)
    ** where n only has as factors 2, 3, 5 and 7
    */
    if (n % (11 * 13) == 0) return 0 ; /* No good */

    return is_2357 (n)	|| ((n % 11 == 0) && is_2357 (n / 11))
                        || ((n % 13 == 0) && is_2357 (n / 13)) ;
}

int main
    (
    int ac,
    char *av[]
    )
{
    if (ac != 6)
        {
        printf("usage: %s [win secs] [step secs] [export image height] [cfg file] [weights file]\n", av[0]);
        exit(1);
        }

    int i = 0;
    int j = 0;
    int k = 0;
    const int SAMPLERATE = 16000;
    PaStreamParameters input_parameters;
    PaError err;
    PaStream *stream;

    err = Pa_Initialize();
    if (err != paNoError)
        {
        Pa_Terminate();
        return(EXIT_FAILURE);
        }

    input_parameters.device = Pa_GetDefaultInputDevice();
    if (input_parameters.device == paNoDevice)
        {
        puts("No default input device");
        Pa_Terminate();
        return(EXIT_FAILURE);
        }

    input_parameters.channelCount = 1;
    input_parameters.sampleFormat = paFloat32;
    input_parameters.suggestedLatency = Pa_GetDeviceInfo(input_parameters.device)->defaultLowInputLatency;
    input_parameters.hostApiSpecificStreamInfo = NULL;
    err = Pa_OpenStream(&stream, &input_parameters, NULL, SAMPLERATE, 1024, paClipOff, NULL, NULL);
    if (err != paNoError)
        {
        Pa_Terminate();
        return(EXIT_FAILURE);
        }

    err = Pa_StartStream( stream );
    if (err != paNoError)
        {
        Pa_Terminate();
        return(EXIT_FAILURE);
        }

    const double MIN_FREQ = 0.0;
    const double MAX_FREQ = (double)SAMPLERATE / 2.0;
    const double WIN_SECS = atof(av[1]);
    const double STEP_SECS = atof(av[2]);
    const int WIN_SIZE = (int)(WIN_SECS * SAMPLERATE);
    const int STEP_SIZE = (int)(STEP_SECS * SAMPLERATE);
    const int RESERVE_SIZE = WIN_SIZE - STEP_SIZE;
    if (RESERVE_SIZE < 0)
        {
        puts("step size greater than window size");
        return(EXIT_FAILURE);
        }
    const double SPEC_FLOOR_DB = -180.0;
    const double LINEAR_SPEC_FLOOR = pow(10.0, SPEC_FLOOR_DB / 20.0);
    const int PIXEL_WIDTH_PER_SECOND = 100;
    const int SPECTROGRAM_W = PIXEL_WIDTH_PER_SECOND * WIN_SECS;
    const int SPECTROGRAM_H = atoi(av[3]);
    const double MAG_TO_NORMALIZE = 100.0;

    int speclen = SPECTROGRAM_H * (SAMPLERATE / 20 / SPECTROGRAM_H + 1);
    for (i = 0; ; ++i)
        {
        if (is_good_speclen(speclen + i))
            {
            speclen += i;
            break;
            }
        if (speclen - i >= SPECTROGRAM_H && is_good_speclen(speclen - i))
            {
            speclen -= i;
            break;
            }
        }
    spectrum *spec = create_spectrum(speclen, KAISER);
    float **mag_spec = (float **)calloc(SPECTROGRAM_W, sizeof(float *));
    for (i = 0; i < SPECTROGRAM_W ; ++i)
        {
        if ((mag_spec[i] = (float *)calloc(SPECTROGRAM_H, sizeof(float))) == NULL)
            {
            puts("Not enough memory");
            return(EXIT_FAILURE);
            }
        }

    float *clip = (float *)calloc(WIN_SIZE, sizeof(float));
    float *clip_fillin_pos = clip + RESERVE_SIZE;
    float *clip_stepin_pos = clip + STEP_SIZE;
    int frameidx = 0;
    Pa_ReadStream(stream, clip, RESERVE_SIZE);

    cv::Mat im(SPECTROGRAM_H, SPECTROGRAM_W, CV_8UC3);
    unsigned char *imdata = im.data;
    unsigned char colour[3] = {0, 0, 0};

    init_net(av[4], av[5], SPECTROGRAM_W, SPECTROGRAM_H, 3);

    while (1)
        {
        Pa_ReadStream(stream, clip_fillin_pos, STEP_SIZE);
        //printf("%d\n", frameidx);

        for (j = 0; j < SPECTROGRAM_W; ++j)
            {
            int datalen = 2 * speclen;
            double *data = spec->time_domain;
            memset(data, 0, 2 * speclen * sizeof(double));

            int start = (j * WIN_SIZE) / SPECTROGRAM_W - speclen;
            if (start >= 0)
                {
                int copylen = 0;
                if (start + datalen > WIN_SIZE)
                    {
                    copylen = WIN_SIZE - start;
                    }
                else
                    {
                    copylen = datalen;
                    }
                for (i = 0; i < copylen; ++i)
                    {
                    data[i] = clip[i + start]; // float to double
                    }
                }
            else
                {
                start = -start;
                data += start;
                datalen -= start;
                for (i = 0; i < datalen; ++i)
                    {
                    data[i] = clip[i]; // float to double
                    }
                }
            calc_magnitude_spectrum(spec);
            interp_spec(mag_spec[j], SPECTROGRAM_H, spec->mag_spec, speclen, MIN_FREQ, MAX_FREQ, SAMPLERATE);
            }

        // draw spectrogram
        for (j = 0; j < SPECTROGRAM_W; ++j)
            {
            for (k = 0; k < SPECTROGRAM_H; ++k)
                {
                mag_spec[j][k] /= MAG_TO_NORMALIZE;
                mag_spec[j][k] = (mag_spec[j][k] < LINEAR_SPEC_FLOOR) ? SPEC_FLOOR_DB : 20.0 * log10(mag_spec[j][k]);
                get_colour_map_value(mag_spec[j][k], SPEC_FLOOR_DB, colour);
                imdata[((SPECTROGRAM_H - 1 - k) * im.cols + j) * 3] = colour[2];
                imdata[((SPECTROGRAM_H - 1 - k) * im.cols + j) * 3 + 1] = colour[1];
                imdata[((SPECTROGRAM_H - 1 - k) * im.cols + j) * 3 + 2] = colour[0];
                }
            }

        float *netout = run_net(im.data);
        printf("%f, %f\n", netout[0], netout[1]);
        if (netout[0] > 0.9f)
            {
            cv::rectangle(im, cv::Rect(20, 10, SPECTROGRAM_W - 40, SPECTROGRAM_H - 20), cv::Scalar(255, 0, 0), 15, 16);
            cv::putText(im, "dog bark", cv::Point(SPECTROGRAM_W/2-40, SPECTROGRAM_H/2), cv::FONT_HERSHEY_DUPLEX, 1.0, cv::Scalar(255, 0, 0), 2, 16);
            }

        cv::imshow("demo", im);
        unsigned char key = cv::waitKey(10);
        if (key == 27)
            {
            break;
            }

        memcpy(clip, clip_stepin_pos, RESERVE_SIZE * sizeof(float));
        ++frameidx;
        //printf("=========================\n");
        }

    free_net();

    destroy_spectrum(spec);
    for (i = 0; i < SPECTROGRAM_W ; ++i)
        {
        free(mag_spec[i]);
        }
    free(mag_spec);

    err = Pa_CloseStream(stream);
    if( err != paNoError )
        {
        Pa_Terminate();
        return(EXIT_FAILURE);
        }

    Pa_Terminate();
    free(clip);
    return 0;
}

