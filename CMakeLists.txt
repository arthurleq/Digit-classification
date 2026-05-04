cmake_minimum_required(VERSION 3.10)
project(DigitRecognizer)

set(CMAKE_CXX_STANDARD 17)

find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Paths to TensorFlow C API
set(TF_ROOT "/usr/local")
include_directories(${TF_ROOT}/include)
link_directories(${TF_ROOT}/lib)

add_executable(digit_recognizer main.cpp)
target_link_libraries(digit_recognizer
    sfml-graphics sfml-window sfml-system
    tensorflow
)