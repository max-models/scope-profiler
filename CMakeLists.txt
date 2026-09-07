cmake_minimum_required(VERSION 3.15)

project(scope-profiler VERSION 0.5.1 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(SCOPE_PROFILER_BUILD_EXAMPLES "Build the native C example" OFF)
option(SCOPE_PROFILER_ENABLE_LIKWID "Expose the optional LIKWID marker target" OFF)
option(SCOPE_PROFILER_ENABLE_MPI "Expose the optional native MPI wrapper target" OFF)

add_library(scope_profiler_native STATIC
    src/scope_profiler/c/scope_profiler.c
)
add_library(scope-profiler::native ALIAS scope_profiler_native)
set_target_properties(scope_profiler_native PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED YES
    C_EXTENSIONS NO
    EXPORT_NAME native
)
target_include_directories(scope_profiler_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/scope_profiler/c>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/scope-profiler>
)

add_library(scope_profiler_cpp INTERFACE)
add_library(scope-profiler::cpp ALIAS scope_profiler_cpp)
set_target_properties(scope_profiler_cpp PROPERTIES EXPORT_NAME cpp)
target_compile_features(scope_profiler_cpp INTERFACE cxx_std_11)
target_link_libraries(scope_profiler_cpp INTERFACE scope_profiler_native)

add_library(scope_profiler_header_only INTERFACE)
add_library(scope-profiler::header-only ALIAS scope_profiler_header_only)
set_target_properties(scope_profiler_header_only PROPERTIES EXPORT_NAME header-only)
target_compile_features(scope_profiler_header_only INTERFACE cxx_std_17)
target_compile_definitions(scope_profiler_header_only INTERFACE SP_HEADER_ONLY)
target_include_directories(scope_profiler_header_only INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/scope_profiler/c>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/scope-profiler>
)

if(SCOPE_PROFILER_ENABLE_LIKWID)
    find_path(SCOPE_PROFILER_LIKWID_INCLUDE_DIR likwid-marker.h REQUIRED)
    find_library(SCOPE_PROFILER_LIKWID_LIBRARY NAMES likwid REQUIRED)
    add_library(scope_profiler_likwid INTERFACE)
    add_library(scope-profiler::likwid ALIAS scope_profiler_likwid)
    set_target_properties(scope_profiler_likwid PROPERTIES EXPORT_NAME likwid)
    target_compile_definitions(scope_profiler_likwid INTERFACE
        SP_USE_LIKWID LIKWID_PERFMON
    )
    target_include_directories(scope_profiler_likwid INTERFACE
        "${SCOPE_PROFILER_LIKWID_INCLUDE_DIR}"
    )
    target_link_libraries(scope_profiler_likwid INTERFACE
        "${SCOPE_PROFILER_LIKWID_LIBRARY}"
    )
endif()

if(SCOPE_PROFILER_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    add_library(scope_profiler_mpi INTERFACE)
    add_library(scope-profiler::mpi ALIAS scope_profiler_mpi)
    set_target_properties(scope_profiler_mpi PROPERTIES EXPORT_NAME mpi)
    target_compile_features(scope_profiler_mpi INTERFACE cxx_std_11)
    target_include_directories(scope_profiler_mpi INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/scope_profiler/c>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/scope-profiler>
    )
    target_link_libraries(scope_profiler_mpi INTERFACE MPI::MPI_CXX)
endif()

if(SCOPE_PROFILER_BUILD_EXAMPLES)
    add_executable(scope_profiler_c_example src/scope_profiler/c/example.c)
    target_link_libraries(scope_profiler_c_example PRIVATE scope-profiler::native m)
endif()

install(TARGETS
    scope_profiler_native scope_profiler_cpp scope_profiler_header_only
    EXPORT scope-profilerTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if(TARGET scope_profiler_likwid)
    install(TARGETS scope_profiler_likwid EXPORT scope-profilerTargets)
endif()
if(TARGET scope_profiler_mpi)
    install(TARGETS scope_profiler_mpi EXPORT scope-profilerTargets)
endif()

install(FILES
    src/scope_profiler/c/scope_profiler.h
    src/scope_profiler/c/scope_profiler.hpp
    src/scope_profiler/c/scope_profiler_mpi.hpp
    src/scope_profiler/c/scope_profiler_impl.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/scope-profiler
)
install(EXPORT scope-profilerTargets
    FILE scope-profilerTargets.cmake
    NAMESPACE scope-profiler::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scope-profiler
)

configure_package_config_file(
    cmake/scope-profilerConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/scope-profilerConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scope-profiler
)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/scope-profilerConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/scope-profilerConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/scope-profilerConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/scope-profiler
)
