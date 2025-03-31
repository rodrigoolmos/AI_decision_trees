cmake_minimum_required(VERSION 3.10)

project(laod_and_execute_model)

set(CMAKE_C_STANDARD 99)

set(DIR_SW ${CMAKE_SOURCE_DIR}/C_code_vitis)
set(DIR_HW ${CMAKE_SOURCE_DIR}/linux_drivers)

set(LIB_SOURCES_SW
    ${DIR_SW}/predict.c
    ${DIR_SW}/predict.h
    ${DIR_SW}/diabetes_dataset.h
    ${DIR_SW}/diabetes_model.h
)

set(LIB_SOURCES_HW
    ${DIR_HW}/common/common.c
    ${DIR_HW}/common/common.h
)

add_library(predict_lib STATIC ${LIB_SOURCES_SW})

add_library(common_lib STATIC ${LIB_SOURCES_HW})

set(SOURCES_SW ${DIR_SW}/laod_and_execute_model.c)

add_executable(execute_model_sw ${SOURCES_SW})

if(UNIX AND NOT APPLE)
    set(SOURCES_HW ${DIR_HW}/dma/dma.c)
    add_executable(execute_model_hw_dma ${SOURCES_HW})
endif()

if(UNIX AND NOT APPLE)
    set(SOURCES_HW ${DIR_HW}/memory_mapped/memory_mapped.c)
    add_executable(execute_model_hw_memory_mapped ${SOURCES_HW})
endif()

if(UNIX AND NOT APPLE)
    set(SOURCES_HW ${DIR_HW}/MIG_2_DDR3_DMA/main.c)
    add_executable(execute_model_MIG ${SOURCES_HW})
endif()

target_link_libraries(execute_model_sw PRIVATE predict_lib)
target_link_libraries(execute_model_hw_dma PRIVATE common_lib)
target_link_libraries(execute_model_MIG PRIVATE common_lib)
target_link_libraries(execute_model_hw_memory_mapped PRIVATE common_lib)

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/trained_models)
