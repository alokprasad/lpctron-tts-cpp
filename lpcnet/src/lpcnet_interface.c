#include "lpcnet_interface.h"
#include "arch.h"
#include "lpcnet.h"
#include "freq.h"
#include <stdio.h>
#include <string.h>

static LPCNetState *net;
// static float features[NB_FEATURES];
// static short pcm[FRAME_SIZE];
// static float in_features[NB_BANDS+2];

void init_lpcnet() {
    net = lpcnet_create();
}

void cleanup() {
    lpcnet_destroy(net);
}

void run_lpcnet(float *features_taco, int num_features_taco, void (*pcm_callback)(short *pcm, int pcm_size)) {
    int read_idx = 0;
    while (read_idx < num_features_taco) {
        float features[NB_FEATURES];
        short pcm[FRAME_SIZE];
        float *in_features = features_taco + read_idx;
  
        RNN_COPY(features, in_features, NB_BANDS);
        RNN_CLEAR(&features[18], 18);
        RNN_COPY(features+36, in_features+NB_BANDS, 2);
        lpcnet_synthesize(net, pcm, features, FRAME_SIZE);
       
        pcm_callback(pcm, FRAME_SIZE);
        read_idx += (NB_BANDS + 2);
    }
}