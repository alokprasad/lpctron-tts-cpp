#ifndef LPCNET_INTERFACE_H_
#define LPCNET_INTERFACE_H_

void init_lpcnet();
void run_lpcnet(float *features, int features_size, void (*pcm_callback)(short *pcm, int pcm_size));
void cleanup();

#endif