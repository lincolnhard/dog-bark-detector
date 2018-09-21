# dog-bark-detector

Detect dog bark from CNN based spectrogram classification

## Requirement

Dataset:
[Urbansound](https://urbansounddataset.weebly.com/urbansound.html) (training/testing set)
[ESC-50](https://github.com/karoldvl/ESC-50) (testing set)
[Freiburg 106](http://www.csc.kth.se/~jastork/pages/datasets.html) (training set, negative samples)
3rdparty libraries:
[libsndfile](http://www.mega-nerd.com/libsndfile/#Download)
[fftw3](http://www.fftw.org/download.html)
[opencv](https://github.com/opencv/opencv)
[darknet](https://github.com/pjreddie/darknet)
[portaudio](http://www.portaudio.com/download.html)



## Preprocessing

```
# To slice audio files into 2 secs duration clips, and store them into new .wav files
# Note that this will take up to 25 GB on your disk

python preprocessing.py --urbansound_dir [folder UrbanSound] --esc50_dir [folder ESC-50] --kitchen106_dir [folder building_106_kitchen/building_106_kitchen]

# Generate spectrograms
# Edit makefile, specify your darknet header/lib location

make
./create_spectrogram [folder UrbanSound] [folder ESC-50] [folder building_106_kitchen/building_106_kitchen]
```


## Training

Edit [Your darknet folder]/examples/classifier.c
In function **void train_classifier(...)**, change **args.type** from **CLASSIFICATION_DATA** to **OLD_CLASSIFICATION_DATA**, and rebuild darknet, since we don't want darknet augments input data for us.

Edit **cfg/dogbark.data**, specify your training and validation set list.
```
# Training
./darknet classifier train cfg/dogbark.data cfg/dogbark.cfg
# Testing
./darknet classifier valid cfg/dogbark.data cfg/dogbark.test.cfg backup/dogbark.backup
```
#### Urbansound top-1 accuracy should be around 95%
#### ESC-50 Top-1 accuracy should be around 80%


## Visualize
Test from file:
```
./classification_from_file [sound file] [win secs] [step secs] [export image height] [cfg file] [weights file]
# For example:
#./classification_from_file sound/dogtest.wav 2.8 0.1 200 cfg/dogbark.test.cfg weights/dogbark_32.weights
```
Test from microphone:
```
./classification_from_mic [win secs] [step secs] [export image height] [cfg file] [weights file]
# For example:
./classification_from_mic 2.5 0.02 200 cfg/dogbark.test.cfg weights/dogbark_32.weights
```

![peek 2018-09-21 14-14](https://user-images.githubusercontent.com/16308037/45864322-3c63d400-bdac-11e8-836b-f0ab147f532b.gif)
