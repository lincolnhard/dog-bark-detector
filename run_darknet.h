#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#ifdef __cplusplus
extern "C"{
#endif

void init_net
    (
    const char *cfgfile,
    const char *weightfile,
    const int imw,
    const int imh,
    const int imch
    );

float *run_net
    (
    unsigned char *indata
    );

void free_net
    (
    void
    );

#ifdef __cplusplus
}
#endif

#endif // RUN_DARKNET_H

