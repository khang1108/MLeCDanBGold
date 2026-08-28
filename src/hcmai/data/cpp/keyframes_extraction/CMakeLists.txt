cmake_minimum_required(VERSION 3.20)
project(hcmai_keyframes_extractor VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
    src/disk_guard.cpp
    src/extractor.cpp
    src/ffmpeg.cpp
    src/frame_index.cpp
    src/jsonl.cpp
    src/process.cpp
    src/state.cpp
    src/timestamp_sampler.cpp
)
target_include_directories(keyframes_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
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

add_executable(test_state tests/test_state.cpp)
target_link_libraries(test_state PRIVATE keyframes_core)
target_compile_options(test_state PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_jsonl tests/test_jsonl.cpp)
target_link_libraries(test_jsonl PRIVATE keyframes_core)
target_compile_options(test_jsonl PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_frame_index tests/test_frame_index.cpp)
target_link_libraries(test_frame_index PRIVATE keyframes_core)
target_compile_options(test_frame_index PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_timestamp_sampler tests/test_timestamp_sampler.cpp)
target_link_libraries(test_timestamp_sampler PRIVATE keyframes_core)
target_compile_options(test_timestamp_sampler PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_ffmpeg tests/test_ffmpeg.cpp)
target_link_libraries(test_ffmpeg PRIVATE keyframes_core)
target_compile_options(test_ffmpeg PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_disk_guard tests/test_disk_guard.cpp)
target_link_libraries(test_disk_guard PRIVATE keyframes_core)
target_compile_options(test_disk_guard PRIVATE -Wall -Wextra -Wpedantic -Werror)

add_executable(test_extractor_smoke tests/test_extractor_smoke.cpp)
target_link_libraries(test_extractor_smoke PRIVATE keyframes_core)
target_compile_options(test_extractor_smoke PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_dependencies(test_extractor_smoke keyframe_extractor)
target_compile_definitions(test_extractor_smoke PRIVATE
    KEYFRAME_EXTRACTOR_PATH="$<TARGET_FILE:keyframe_extractor>"
)

add_executable(test_publication tests/test_publication.cpp)
target_link_libraries(test_publication PRIVATE keyframes_core)
target_compile_options(test_publication PRIVATE -Wall -Wextra -Wpedantic -Werror)
add_dependencies(test_publication keyframe_extractor)
target_compile_definitions(test_publication PRIVATE
    KEYFRAME_EXTRACTOR_PATH="$<TARGET_FILE:keyframe_extractor>"
)

add_test(NAME process_unit COMMAND test_process)
add_test(NAME state_unit COMMAND test_state)
add_test(NAME jsonl_unit COMMAND test_jsonl)
add_test(NAME frame_index_unit COMMAND test_frame_index)
add_test(NAME timestamp_sampler_unit COMMAND test_timestamp_sampler)
add_test(NAME ffmpeg_unit COMMAND test_ffmpeg)
add_test(NAME disk_guard_unit COMMAND test_disk_guard)
add_test(NAME extractor_smoke COMMAND test_extractor_smoke)
add_test(NAME publication_unit COMMAND test_publication)
add_test(NAME keyframe_extractor_version COMMAND keyframe_extractor --version)
set_tests_properties(
    keyframe_extractor_version
    PROPERTIES PASS_REGULAR_EXPRESSION "hcmai-keyframes-extractor/0.1.0"
)
