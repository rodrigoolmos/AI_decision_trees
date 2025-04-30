cmake_minimum_required(VERSION 3.10)

project(laod_and_execute_model)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_C_STANDARD 99)

set(DIR_SW ${CMAKE_SOURCE_DIR}/C_code_vitis)
set(DIR_OPENCL ${CMAKE_SOURCE_DIR}/openCL)
set(DIR_HW ${CMAKE_SOURCE_DIR}/linux_drivers)

set(OpenMP_C_FLAGS "-fopenmp")
set(OpenMP_C_LIB_NAMES "gomp")

find_package(OpenCL QUIET)
find_package(OpenMP REQUIRED)

if(OpenCL_FOUND)
    set(SOURCES_OPENCL ${DIR_OPENCL}/main.c)
    add_executable(execute_model_openCL ${SOURCES_OPENCL})
    target_link_libraries(execute_model_openCL PRIVATE OpenCL::OpenCL common_lib)
endif()

set(LIB_SOURCES_SW
    ${DIR_SW}/predict.c
    ${DIR_SW}/predict.h
    ${DIR_SW}/train.c
    ${DIR_SW}/train.h
)

set(LIB_SOURCES_HW
    ${DIR_HW}/common/common.c
    ${DIR_HW}/common/common.h
    ${DIR_HW}/burst_drivers/burst.c
    ${DIR_HW}/burst_drivers/burst.h
    ${DIR_HW}/train/train.c
    ${DIR_HW}/train/train.h
)

add_library(predict_lib STATIC ${LIB_SOURCES_SW})

add_library(common_lib STATIC ${LIB_SOURCES_HW})

set(SOURCES_SW ${DIR_SW}/train_export_model.c)

add_executable(execute_model_sw ${SOURCES_SW})

if(UNIX AND NOT APPLE)

    set(SOURCES_HW ${DIR_HW}/use_example.c)
    add_executable(execute_model_burst ${SOURCES_HW})

    target_link_libraries(execute_model_burst PRIVATE common_lib m OpenMP::OpenMP_C)

endif()

target_link_libraries(execute_model_sw PRIVATE predict_lib m OpenMP::OpenMP_C)


file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/trained_models)
