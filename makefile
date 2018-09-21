CCL += gcc
CXXCL += g++
DARKNET_DIR += Your_darknet_folder_location
# ex. DARKNET_DIR += /home/lincolnhard/Documents/darknet

INCLUDES += `pkg-config --cflags sndfile`
INCLUDES += `pkg-config --cflags fftw3`
INCLUDES += `pkg-config --cflags opencv`
INCLUDES += `pkg-config --cflags portaudio-2.0`
INCLUDES += -I$(DARKNET_DIR)/include

CFLAGS += -Ofast
CXXFLAGS += -Ofast -std=c++11

LDFLAGS += `pkg-config --libs sndfile`
LDFLAGS += `pkg-config --libs fftw3`
LDFLAGS += `pkg-config --libs opencv`
LDFLAGS += `pkg-config --libs portaudio-2.0`
LDFLAGS += -lcairo
LDFLAGS += -L$(DARKNET_DIR)
LDFLAGS += -ldarknet

OBJS += spectrum.o window.o common.o run_darknet.o
EXECOBJ1 += create_spectrogram.o
EXECOBJ2 += classification_from_mic.o
EXECOBJ3 += classification_from_file.o

all: $(EXECOBJ1) $(EXECOBJ2) $(EXECOBJ3) $(OBJS)
	$(CCL) -o create_spectrogram $(EXECOBJ1) $(OBJS) $(LDFLAGS)
	$(CXXCL) -o classification_from_mic $(EXECOBJ2) $(OBJS) $(LDFLAGS)
	$(CXXCL) -o classification_from_file $(EXECOBJ3) $(OBJS) $(LDFLAGS)

%.o: %.c
	$(CCL) -c -pipe $(CFLAGS) $(DEFINES) $(INCLUDES) $<

%.o: %.cpp
	$(CXXCL) -c -pipe $(CXXFLAGS) $(DEFINES) $(INCLUDES) $<

clean:
	rm *.o; rm create_spectrogram classification_from_mic classification_from_file
