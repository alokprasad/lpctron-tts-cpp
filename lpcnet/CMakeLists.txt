cmake_minimum_required(VERSION 3.10.2)

add_library(lpcnet SHARED
  src/lpcnet.c
  src/lpcnet_interface.c
  src/nnet.c 
  src/nnet_data.c 
  src/freq.c
  src/kiss_fft.c 
  src/pitch.c
  src/celt_lpc.c)


target_include_directories(lpcnet 
PRIVATE include interface
INTERFACE interface)

set (CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function  -mavx2 -mfma  -O3 -DTACOTRON2")
set (CMAKE_CXX_FLAGS "-Wall -W -Wextra -Wno-unused-function  -mavx2 -mfma  -O3 -DTACOTRON2")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

#set_source_files_properties(${SOURCES}
#PROPERTIES COMPILE_FLAGS "-O3")

#X86
#set (CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function  -mavx2 -mfma  -O3 -DTACOTRON2")

#ARM
#ffast-math requires #define FLOAT_APRROX 1 in LPCNET/src/arch.h
#set (CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function  -ffast-math -Ofast -DTACOTRON2 -march=armv8-a  -mtune=cortex-a57")
#set (CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function  -ffast-math -Ofast -O3 -DTACOTRON2 -mfpu=neon -march=armv8-a  -mtune=cortex-a57")



target_compile_definitions(lpcnet PRIVATE TACOTRON2)

target_link_libraries(lpcnet m)
