cmake_minimum_required(VERSION 3.20)
project(hcmai_keyframes_extractor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale
)
pkg_check_modules(JSONC REQUIRED IMPORTED_TARGET json-c)

add_library(keyframes_core STATIC
    src/config.cpp
    src/extractor.cpp
    src/ffmpeg.cpp
    src/frame_index.cpp
    src/jsonl.cpp
    src/process.cpp
    src/state.cpp
    src/timestamp_sampler.cpp
)
target_include_directories(keyframes_core PUBLIC include)
target_link_libraries(keyframes_core PUBLIC PkgConfig::FFMPEG PkgConfig::JSONC)
target_compile_options(keyframes_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(keyframe_extractor src/main.cpp)
target_link_libraries(keyframe_extractor PRIVATE keyframes_core)
target_compile_options(keyframe_extractor PRIVATE -Wall -Wextra -Wpedantic -Werror)

include(CTest)
enable_testing()

add_executable(test_process tests/test_process.cpp)
target_link_libraries(test_process PRIVATE keyframes_core)
target_compile_options(test_process PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_jsonl tests/test_jsonl.cpp)
target_link_libraries(test_jsonl PRIVATE keyframes_core)
target_compile_options(test_jsonl PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_test(NAME process_unit COMMAND test_process)
add_test(NAME jsonl_unit COMMAND test_jsonl)
add_test(NAME keyframe_extractor_version COMMAND keyframe_extractor --version)
set_tests_properties(
    keyframe_extractor_version
    PROPERTIES PASS_REGULAR_EXPRESSION "hcmai-keyframes-extractor/0.1.0"
)
