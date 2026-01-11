cmake_minimum_required(VERSION 3.18)

project(FIRE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)


add_executable(FireDB
        main.cpp
        src/core/slab.h
        src/core/gpu.h
        src/core/gpu.h
        src/core/slab.h
)


set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)

target_compile_options(FireDB PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native>
)

target_link_libraries(FireDB PRIVATE CUDA::cudart CUDA::cublas)