# DwarfStar 4 - Windows build (MSVC + CUDA)
#
# This CMakeLists.txt is the Windows-only entry point.  Linux and macOS users
# continue to use the existing Makefile (it does not invoke CMake).  On
# Windows, the Makefile is unusable (uname, -march, pthreads, mmap, ...), so
# we provide a parallel build path that targets the Microsoft toolchain and
# CUDA on Windows.
#
# Required:
#   - Windows 10 build 19041 (version 2004) or newer
#   - Visual Studio 2022 with "Desktop development with C++"
#   - CUDA Toolkit 12.8 or newer (for sm_120 Blackwell)
#   - NVIDIA driver 570 or newer
#   - CMake 3.24 or newer
#
# Build:
#   cmake -B build -G "Visual Studio 17 2022" -A x64 ^
#       -DCMAKE_BUILD_TYPE=Release ^
#       -DCMAKE_CUDA_ARCHITECTURES=120
#   cmake --build build --config Release --target ds4-server ds4-bench ds4-eval

cmake_minimum_required(VERSION 3.24)
project(ds4 LANGUAGES C CXX CUDA)

if(NOT WIN32)
    message(FATAL_ERROR
        "This CMakeLists.txt is the Windows-only build entry point.  "
        "On Linux/macOS, use the Makefile: 'make cuda-generic' or 'make'.")
endif()

# Default to the RTX 5070 / Blackwell consumer target; let the user widen it.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDAToolkit REQUIRED)

if(CUDAToolkit_VERSION VERSION_LESS "12.8")
    message(WARNING
        "CUDA Toolkit ${CUDAToolkit_VERSION} found, but Blackwell sm_120 "
        "(RTX 5070/5080/5090) requires CUDA 12.8 or newer.  Build will "
        "continue but the resulting binary may not run on a 5070.")
endif()

add_compile_definitions(
    _WIN32_WINNT=0x0A00
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    _CRT_SECURE_NO_WARNINGS
)

# Equivalent to Makefile's -O3 -ffast-math.  /wd disables a few warning
# categories that fire heavily from this codebase's C99 idioms without
# indicating real problems (size_t<>int truncation, possible-loss).
add_compile_options(
    $<$<COMPILE_LANGUAGE:C>:/O2>
    $<$<COMPILE_LANGUAGE:C>:/fp:fast>
    $<$<COMPILE_LANGUAGE:C>:/W3>
    $<$<COMPILE_LANGUAGE:C>:/wd4244>
    $<$<COMPILE_LANGUAGE:C>:/wd4267>
    $<$<COMPILE_LANGUAGE:C>:/wd4101>
    $<$<COMPILE_LANGUAGE:C>:/wd4146>
    $<$<COMPILE_LANGUAGE:CXX>:/O2>
    $<$<COMPILE_LANGUAGE:CXX>:/fp:fast>
    $<$<COMPILE_LANGUAGE:CXX>:/W3>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4244>
    $<$<COMPILE_LANGUAGE:CXX>:/wd4267>
)

# Equivalent to Makefile's NVCCFLAGS = -O3 --use_fast_math -Xcompiler $(NATIVE_CPU_FLAG)
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/O2>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/fp:fast>
    # CUDA 13.x CCCL headers refuse the MSVC traditional preprocessor.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# Core static library shared by all three binaries.  Contains the model
# loader, the CUDA backend, and the Windows platform shims.
add_library(ds4_core STATIC
    ds4.c
    ds4_cuda.cu
    ds4_platform_win.c
)
set_target_properties(ds4_core PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE   ON
)
target_include_directories(ds4_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ds4_core PUBLIC
    CUDA::cudart
    CUDA::cublas
    ws2_32
    bcrypt
)

# Three executables.  No ds4.exe in this phase: that needs a linenoise port.
add_executable(ds4-server  ds4_server.c rax.c)
add_executable(ds4-bench   ds4_bench.c)
add_executable(ds4-eval    ds4_eval.c)

foreach(t ds4-server ds4-bench ds4-eval)
    target_link_libraries(${t} PRIVATE ds4_core)
    # Use CUDA as the link driver so device symbols resolve cleanly.
    set_target_properties(${t} PROPERTIES LINKER_LANGUAGE CUDA)
endforeach()

# CUDA runtime DLLs (cudart64_*.dll, cublas64_*.dll, …) are loaded from the
# %PATH% that the CUDA Toolkit installer prepends.  We deliberately do NOT
# copy them next to the .exe: TARGET_RUNTIME_DLLS is empty for our import
# libs and the empty-list copy_if_different fails the post-build step.
