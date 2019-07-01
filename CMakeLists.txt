
##LINUX
#Build just change require CMakeLists*.txt file in include and use cmake .. and make from build.
#TF git rev-parse HEAD - > 70be2de5dd7743a571fffc6a2cbc2905bd18063a 
#But it was buit using 19c 
#sudo apt-get install autoconf automake libtool curl make g++ unzip zlib1g-dev git python
#tensorflow/contrib/makefile/build_all_linux.sh
#git clone https://github.com/tensorflow/tensorflow.git
#cd tensorflow
#git checkout -b r1.14
#git pull
#tensorflow/contrib/makefile/build_all_linux.sh

##ANDROID

# ANdroid - Compile Tensorflow with ndk14b ( dockerfile ) and TTS with NDK15b 
# #cmake -DCMAKE_TOOLCHAIN_FILE=${NDK_ROOT}/build/cmake/android.toolchain.cmake     -DANDROID_ABI=arm64-v8a   -DTARGET_PLATFORM=android -DTARGET_ARCHITECTURE=arm64 -DNDK_PATH=${NDK_ROOT} ..
# tensorflow 1.13 compiled with ndk14b ( check docker file , grep ANDROID_NDK_FILENAME) and tts with ndk15b

cmake_minimum_required(VERSION 3.10)
#include(CMakeLists-static-linux.txt)
include(CMakeLists-static-android.txt)
