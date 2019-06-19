This repo contains c/c++ version of Tacotron2 and LPCNet .
## Prerequisite
   Tensorflow bazel build

## Merge Model( as github has limit for 100mb)
cd model
bash split_merge - > generates inference_model.pb

## Build and Run
1. git clone the repo
2. mkdir build && cd build
3. make
4. ./lpctron_cc sample.txt
5. ffplay -f s16le -ar 16k -ac 1 output.pcm



