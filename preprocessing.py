import os
from pydub import AudioSegment
from pydub.generators import WhiteNoise

def main(args):
    urbansound_folder = args.urbansound_dir
    urbansound_dogbark_data_folder = urbansound_folder + os.sep + 'data/dog_bark'
    urbansound_graph_folder = urbansound_folder + os.sep + 'graph'
    urbansound_dogbark_graph_folder = urbansound_graph_folder + os.sep + 'positive'
    urbansound_other_graph_folder = urbansound_graph_folder + os.sep + 'negative'
    esc50_folder = args.esc50_dir
    esc50_graph_folder = esc50_folder + os.sep + 'graph'
    esc50_dogbark_graph_folder = esc50_graph_folder + os.sep + 'positive'
    esc50_other_graph_folder = esc50_graph_folder + os.sep + 'negative'
    building_106_kitchen_folder = args.kitchen106_dir
    building_106_kitchen_graph_folder = building_106_kitchen_folder + os.sep + 'graph'
    building_106_kitchen_other_graph_folder = building_106_kitchen_graph_folder + os.sep + 'negative'

    print esc50_dogbark_graph_folder
    print building_106_kitchen_other_graph_folder

    if not os.path.exists(urbansound_graph_folder):
        os.mkdir(urbansound_graph_folder, 0755)
    if not os.path.exists(urbansound_dogbark_graph_folder):
        os.mkdir(urbansound_dogbark_graph_folder, 0755)
    if not os.path.exists(urbansound_other_graph_folder):
        os.mkdir(urbansound_other_graph_folder, 0755)
    if not os.path.exists(esc50_graph_folder):
        os.mkdir(esc50_graph_folder, 0755)
    if not os.path.exists(esc50_dogbark_graph_folder):
        os.mkdir(esc50_dogbark_graph_folder, 0755)
    if not os.path.exists(esc50_other_graph_folder):
        os.mkdir(esc50_other_graph_folder, 0755)
    if not os.path.exists(building_106_kitchen_graph_folder):
        os.mkdir(building_106_kitchen_graph_folder, 0755)
    if not os.path.exists(building_106_kitchen_other_graph_folder):
        os.mkdir(building_106_kitchen_other_graph_folder, 0755)

    urbansound_other_data_folders = [urbansound_folder + os.sep + 'data/air_conditioner',
                                     urbansound_folder + os.sep + 'data/car_horn', \
                                     urbansound_folder + os.sep + 'data/children_playing',
                                     urbansound_folder + os.sep + 'data/drilling', \
                                     urbansound_folder + os.sep + 'data/engine_idling',
                                     urbansound_folder + os.sep + 'data/gun_shot', \
                                     urbansound_folder + os.sep + 'data/jackhammer',
                                     urbansound_folder + os.sep + 'data/siren', \
                                     urbansound_folder + os.sep + 'data/street_music']

    building_106_kitchen_other_data_folders = [building_106_kitchen_folder + os.sep + 'training_segments/bag', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/blender',
                                               building_106_kitchen_folder + os.sep + 'training_segments/cornflakes_bowl', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/cornflakes_eating',
                                               building_106_kitchen_folder + os.sep + 'training_segments/cup', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/dish_washer',
                                               building_106_kitchen_folder + os.sep + 'training_segments/electric_razor', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/flatware_sorting',
                                               building_106_kitchen_folder + os.sep + 'training_segments/food_processor', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/hair_dryer',
                                               building_106_kitchen_folder + os.sep + 'training_segments/microwave', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/microwave_bell',
                                               building_106_kitchen_folder + os.sep + 'training_segments/microwave_door', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/plates_sorting',
                                               building_106_kitchen_folder + os.sep + 'training_segments/stirring_cup', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/toaster_up_down',
                                               building_106_kitchen_folder + os.sep + 'training_segments/toilet_button', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/toilet_flush',
                                               building_106_kitchen_folder + os.sep + 'training_segments/tooth', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/vacuum_cleaner',
                                               building_106_kitchen_folder + os.sep + 'training_segments/washing_machine', \
                                               building_106_kitchen_folder + os.sep + 'training_segments/water_boiler',
                                               building_106_kitchen_folder + os.sep + 'training_segments/water_tap']

    SECOND_MS = 1000
    SEGMENT_MS = 2000
    ASSIGNED_SAMPLERATE = 44100
    ESC50_AUDIO_START_POS = 500
    POSITIVE_SAMPLE_DB_TH = -40.0

    print 'creating positive training set ..'
    idx = 0
    for file in os.listdir(urbansound_dogbark_data_folder):
        filename, extension = os.path.splitext(file)
        if extension == '.wav' or extension == '.ogg' or extension == '.mp3' or extension == '.flac' or extension == '.aif' or extension == '.aiff':
            # open sound file
            audiopath = urbansound_dogbark_data_folder + os.sep + file
            print audiopath
            audio = AudioSegment.from_file(audiopath).set_frame_rate(ASSIGNED_SAMPLERATE).set_channels(1).set_sample_width(2)[:]
            # open csv file
            csvpath = urbansound_dogbark_data_folder + os.sep + filename + '.csv'
            csv = open(csvpath, 'r')
            lines = csv.readlines()
            for line in lines:
                start = float(line.split(',')[0]) * SECOND_MS
                end = float(line.split(',')[1]) * SECOND_MS
                chunk1 = (end - start) / 10
                current = start
                while 1:
                    outfile = urbansound_dogbark_graph_folder + os.sep + str(idx) + '_dogbark.wav'
                    idx += 1
                    audioclip = audio[current:current + SEGMENT_MS]
                    if len(audioclip) != SEGMENT_MS:
                        lack = SEGMENT_MS - len(audioclip) + 100  # 100 for default crossfade
                        noiseclip = WhiteNoise().to_audio_segment(duration=lack, volume=-50)
                        lastclip = audioclip.append(noiseclip)
                        if lastclip.dBFS > POSITIVE_SAMPLE_DB_TH:
                            lastclip.export(outfile, format='wav')
                        break
                    else:
                        if audioclip.dBFS > POSITIVE_SAMPLE_DB_TH:
                            audioclip.export(outfile, format='wav')
                    current += SEGMENT_MS
                    chunk2 = end - current
                    if chunk2 < chunk1:
                        break
                # if current > end:
                # break
            csv.close()

    print 'creating negative training set ..'
    idx = 0
    for other_data_folder in urbansound_other_data_folders:
        for file in os.listdir(other_data_folder):
            filename, extension = os.path.splitext(file)
            if extension == '.wav' or extension == '.ogg' or extension == '.mp3' or extension == '.flac' or extension == '.aif' or extension == '.aiff':
                # open sound file
                audiopath = other_data_folder + os.sep + file
                print audiopath
                try:
                    audio = AudioSegment.from_file(audiopath).set_frame_rate(ASSIGNED_SAMPLERATE).set_channels(
                        1).set_sample_width(2)[:]
                    num_segment = len(audio) / SEGMENT_MS
                    for i in range(0, num_segment):
                        if i % 4 == 0:  # less sample :)
                            outfile = urbansound_other_graph_folder + os.sep + str(idx) + '_other.wav'
                            idx += 1
                            audio[i * SEGMENT_MS: (i + 1) * SEGMENT_MS].export(outfile, format='wav')
                except:
                    print 'failed to load this one ^^^^^'

    print 'creating test set ..'
    idx = 0
    csvpath = esc50_folder + os.sep + 'meta' + os.sep + 'esc50.csv'
    csv = open(csvpath, 'r')
    lines = csv.readlines()
    for line in lines[1:]:
        filename = line.split(',')[0]
        audiopath = esc50_folder + os.sep + 'audio' + os.sep + filename
        print audiopath
        audio = AudioSegment.from_file(audiopath)[:]
        audio = audio.set_frame_rate(ASSIGNED_SAMPLERATE)
        audio = audio.set_channels(1)
        if line.split(',')[3] == 'dog':
            outfile = esc50_dogbark_graph_folder + os.sep + str(idx) + '_dogbark.wav'
        else:
            outfile = esc50_other_graph_folder + os.sep + str(idx) + '_other.wav'
        idx += 1
        audio[ESC50_AUDIO_START_POS: ESC50_AUDIO_START_POS + SEGMENT_MS].export(outfile, format='wav')
    csv.close()

    print 'creating more negative samples'
    idx = 0
    for other_data_folder in building_106_kitchen_other_data_folders:
        for file in os.listdir(other_data_folder):
            filename, extension = os.path.splitext(file)
            if extension == '.wav' or extension == '.ogg' or extension == '.mp3' or extension == '.flac' or extension == '.aif' or extension == '.aiff':
                # open sound file
                audiopath = other_data_folder + os.sep + file
                print audiopath
                try:
                    audio = AudioSegment.from_file(audiopath).set_frame_rate(ASSIGNED_SAMPLERATE).set_channels(
                        1).set_sample_width(2)[:]
                    outfile = building_106_kitchen_other_graph_folder + os.sep + str(idx) + '_other.wav'
                    idx += 1
                    audio[0: SEGMENT_MS].export(outfile, format='wav')
                except:
                    print 'failed to load this one ^^^^^'




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--urbansound_dir', '-u', dest='urbansound_dir', required=True)
    parser.add_argument('--esc50_dir', '-e', dest='esc50_dir', required=True)
    parser.add_argument('--kitchen106_dir', '-k', dest='kitchen106_dir', required=True)
    args = parser.parse_args()
    main(args)
