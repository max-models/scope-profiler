cmake_minimum_required(VERSION 3.15)
project(scope-profiler-consumer LANGUAGES C CXX)

option(SCOPE_PROFILER_USE_FETCHCONTENT "Fetch scope-profiler from Git" OFF)
set(SCOPE_PROFILER_SOURCE_DIR "" CACHE PATH "Use a local source tree with FetchContent")

if(SCOPE_PROFILER_USE_FETCHCONTENT)
    include(FetchContent)
    if(SCOPE_PROFILER_SOURCE_DIR)
        FetchContent_Declare(scope-profiler SOURCE_DIR "${SCOPE_PROFILER_SOURCE_DIR}")
    else()
        FetchContent_Declare(scope-profiler
            GIT_REPOSITORY https://github.com/max-models/scope-profiler.git
            GIT_TAG devel
            GIT_SHALLOW TRUE
        )
    endif()
    FetchContent_MakeAvailable(scope-profiler)
else()
    find_package(scope-profiler CONFIG REQUIRED)
endif()

add_executable(profiled-app main.cpp)
target_link_libraries(profiled-app PRIVATE scope-profiler::header-only)

add_executable(profiled-app-compiled main.cpp)
target_link_libraries(profiled-app-compiled PRIVATE scope-profiler::cpp)

add_executable(profiled-native native.c)
target_link_libraries(profiled-native PRIVATE scope-profiler::native)

if(TARGET scope-profiler::mpi)
    add_executable(profiled-mpi mpi.cpp)
    target_link_libraries(profiled-mpi PRIVATE
        scope-profiler::cpp
        scope-profiler::mpi
    )
endif()
