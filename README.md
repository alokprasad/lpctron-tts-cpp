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



## Requirment

* TensorFlow r1.8+
* Ubuntu 16.04
* C++ compiler + cmake

## Compiling Tensorflow 1.13 

COmpilation of this projects requires compiling tensorflow from source as it uses some headers files that are generated 
during compilation of tensorflow ( bazel compilation)

*  Install  Bazel 0.24 

```
wget https://github.com/bazelbuild/bazel/releases/download/0.24.0/bazel-0.24.0-installer-linux-x86_64.sh
bash bazel-0.24.0-installer-linux-x86_64.sh
```

*  Install Tensorflow

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.13
./configure ( i used CPU version)
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel build //tensorflow:libtensorflow_cc.so
bazel build //tensorflow:libtensorflow.so

More optimizaed build for the x86 platform without cuda
bazel build --config=opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow_cc.so
bazel build --config=opt --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-msse4.2 //tensorflow:libtensorflow.so

```

* Native Compilation of TensorFlow on Aarch64( PX2)

```
sudo apt-get install openjdk-8-jdk
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip
wget https://github.com/bazelbuild/bazel/releases/download/0.21.0/bazel-0.21.0-dist.zip
unzip 
./compile.sh
sudo cp output/bazel /usr/local/bin

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
git checkout r1.13
./configure ( i used CPU version)

bazel build -c opt --jobs=4 --verbose_failures  --copt="-funsafe-math-optimizations" --copt="-O3" --copt="-ftree-vectorize" --copt="-fomit-frame-pointer" //tensorflow:libtensorflow_cc.so
```

* Note : if you abort or build failure error , increase swap space . ( OOM killer message in dmesg)


*  ABSL library

```
cd /usr/local/inclue
git clone https://github.com/abseil/abseil-cpp.git
mv abseil-cpp/absl .
```

## Tensorflow compilation Links
https://gist.github.com/EKami/9869ae6347f68c592c5b5cd181a3b205

http://wangkejie.me/2018/03/01/tensorflow-cplusplus-installation/

https://tuanphuc.github.io/standalone-tensorflow-cpp/

https://gist.github.com/EKami/9869ae6347f68c592c5b5cd181a3b205 ( For ARM Linux)

https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-10-0-for-python-on-ubuntu/2/

https://medium.com/@sometimescasey/building-tensorflow-from-source-for-sse-avx-fma-instructions-worth-the-effort-fbda4e30eec3

The major revision of the Android NDK referenced by android_ndk_repository rule 'androidndk' is 19. The major revisions supported by Bazel are [10, 11, 12, 13, 14, 15, 16, 17, 18]. Bazel will attempt to treat the NDK as if it was r18. This may cause compilation and linkage problems. Please download a supported NDK version.


