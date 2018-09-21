#include "darknet.h"

static network *net;
static image im;
static image crop;
static int srcw;
static int srch;
static int srcch;

void init_net
    (
    const char *cfgfile,
    const char *weightfile,
    const int imw,
    const int imh,
    const int imch
    )
{
    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    im = make_image(imw, imh, imch);
    srcw = imw;
    srch = imh;
    srcch = imch;
}

float *run_net
    (
    unsigned char *indata
    )
{
    int i = 0;
    int j = 0;
    int k = 0;

    for(i = 0; i < srch; ++i)
        {
        for(k= 0; k < srcch; ++k)
            {
            for(j = 0; j < srcw; ++j)
                {
                im.data[k * srcw * srch + i * srcw + j] = indata[i * srcch * srcw + j * srcch + k] / 255.;
                }
            }
        }
    rgbgr_image(im);
    crop = center_crop_image(im, net->w, net->h);
    float *pred = network_predict(net, crop.data);
    return pred;
}

void free_net
    (
    void
    )
{
    //free_network(net);
    free_image(im);
    free_image(crop);
}

