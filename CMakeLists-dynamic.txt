cmake_minimum_required(VERSION 3.10.2)

set (TENSORFLOW_HOME_DIR /media/alok/ws/experiments/tensorflow-1.13.1)

find_library(TENSORFLOW_CC_LIBRARY tensorflow_cc HINTS ${TENSORFLOW_HOME_DIR}/bazel-bin/tensorflow)
find_library(TENSORFLOW_FRAMEWORK_LIBRARY tensorflow_framework HINTS ${TENSORFLOW_HOME_DIR}/bazel-bin/tensorflow)

add_subdirectory(lpcnet)

add_executable(lpctron_cc src/main.cc)

#set (CMAKE_C_FLAGS "-Wall -W -Wextra -Wno-unused-function  -mavx2 -mfma  -O3 -DTACOTRON2")
#set (CMAKE_CXX_FLAGS "-Wall -W -Wextra -Wno-unused-function  -mavx2 -mfma  -O3 -DTACOTRON2")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

target_link_libraries(lpctron_cc 
  ${TENSORFLOW_CC_LIBRARY}
  ${TENSORFLOW_FRAMEWORK_LIBRARY}
  lpcnet)

target_include_directories(lpctron_cc PRIVATE 
  ${TENSORFLOW_HOME_DIR}
  ${TENSORFLOW_HOME_DIR}/bazel-genfiles
  ${TENSORFLOW_HOME_DIR}/bazel-tensorflow-1.13.1/external/eigen_archive
  ${TENSORFLOW_HOME_DIR}/bazel-tensorflow-1.13.1/external/protobuf_archive/src
  ${TENSORFLOW_HOME_DIR}/bazel-tensorflow-1.13.1/external/nsync/public)
