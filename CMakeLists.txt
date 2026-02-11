cmake_minimum_required(VERSION 3.1)

project(zed_open_capture CXX)

############################################################################
# Flags and Policies
cmake_policy(SET CMP0054 NEW)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -s -flto")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -no-pie")

message(STATUS "* Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Processor: ${CMAKE_SYSTEM_PROCESSOR}")

# Raspberry Pi optimization flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message(STATUS "* Optimizing for aarch64 (Raspberry Pi 4 64-bit)")
    add_compile_options(-mtune=cortex-a72 -mcpu=cortex-a72)
    add_definitions(-DEMBEDDED_ARM)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7l")
    message(STATUS "* Optimizing for armv7l (Raspberry Pi 4 32-bit)")
    add_compile_options(-mfloat-abi=hard -mfpu=neon-fp-armv8 -mneon-for-64bits -mtune=cortex-a72)
    add_definitions(-DEMBEDDED_ARM)
endif()

############################################################################
# Options
option(BUILD_VIDEO "Build the ZED Open Capture Video Modules (only for Linux)" ON)
option(BUILD_SENSORS "Build the ZED Open Capture Sensors Modules" ON)
option(BUILD_EXAMPLES "Build the ZED Open Capture examples" ON)
option(DEBUG_CAM_REG "Add functions to log the values of the registers of camera" OFF)

############################################################################
# Source files
set(SRC_VIDEO
    ${PROJECT_SOURCE_DIR}/src/videocapture.cpp
)

set(SRC_SENSORS
    ${PROJECT_SOURCE_DIR}/src/sensorcapture.cpp
)

set(HEADERS_VIDEO
    ${PROJECT_SOURCE_DIR}/include/videocapture.hpp
    ${PROJECT_SOURCE_DIR}/include/defines.hpp
    ${PROJECT_SOURCE_DIR}/include/videocapture_def.hpp
)

set(HEADERS_SENSORS
    ${PROJECT_SOURCE_DIR}/include/sensorcapture.hpp
    ${PROJECT_SOURCE_DIR}/include/defines.hpp
    ${PROJECT_SOURCE_DIR}/include/sensorcapture_def.hpp
)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

############################################################################
# External modules
list(INSERT CMAKE_MODULE_PATH 0 "${PROJECT_SOURCE_DIR}/cmake")

############################################################################
# Find and include dependencies
set(SRC_FULL "")
set(HDR_FULL "")
set(DEP_LIBS "")

if(DEBUG_CAM_REG)
    message(STATUS "* Registers logging enabled")
    add_definitions(-DSENSOR_LOG_AVAILABLE)
endif()

if(BUILD_SENSORS)
    message(STATUS "* Sensors module enabled")
    add_definitions(-DSENSORS_MOD_AVAILABLE)

    find_path(HIDAPI_INCLUDE_DIRS hidapi/hidapi.h)
    find_library(HIDAPI_LIBRARIES NAMES hidapi-hidraw hidapi-libusb)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(HIDAPI DEFAULT_MSG HIDAPI_LIBRARIES HIDAPI_INCLUDE_DIRS)

    if(HIDAPI_FOUND)
        include_directories(${HIDAPI_INCLUDE_DIRS})
        list(APPEND DEP_LIBS ${HIDAPI_LIBRARIES})
    else()
        message(FATAL_ERROR "HIDAPI not found. Please install it before building.")
    endif()

    list(APPEND SRC_FULL ${SRC_SENSORS})
    list(APPEND HDR_FULL ${HEADERS_SENSORS})
endif()

if(BUILD_VIDEO)
    message(STATUS "* Video module enabled")
    add_definitions(-DVIDEO_MOD_AVAILABLE)

    find_path(LibUSB_INCLUDE_DIRS libusb-1.0/libusb.h)
    find_library(LibUSB_LIBRARIES NAMES usb-1.0)

    find_package_handle_standard_args(LIBUSB DEFAULT_MSG LibUSB_LIBRARIES LibUSB_INCLUDE_DIRS)

    if(LIBUSB_FOUND)
        include_directories(${LibUSB_INCLUDE_DIRS})
        list(APPEND DEP_LIBS ${LibUSB_LIBRARIES})
    else()
        message(FATAL_ERROR "LibUSB not found. Please install it before building.")
    endif()

    list(APPEND SRC_FULL ${SRC_VIDEO})
    list(APPEND HDR_FULL ${HEADERS_VIDEO})
endif()

############################################################################
# Library target
add_library(${PROJECT_NAME} SHARED ${SRC_FULL})
target_link_libraries(${PROJECT_NAME} ${DEP_LIBS})

set_target_properties(${PROJECT_NAME} PROPERTIES
    PUBLIC_HEADER "${HDR_FULL}"
)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_PREFIX}/include/zed-open-capture
)

############################################################################
# Examples
if(BUILD_EXAMPLES AND BUILD_VIDEO)
    find_package(OpenMP)
    find_package(OpenCV REQUIRED)

    include_directories(${PROJECT_SOURCE_DIR}/include)
    include_directories(${OpenCV_INCLUDE_DIRS})

    add_executable(main ${PROJECT_SOURCE_DIR}/src/main.cpp)
    set_target_properties(main PROPERTIES PREFIX "")

    if(OpenMP_CXX_FOUND)
        target_link_libraries(main
            OpenMP::OpenMP_CXX
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${DEP_LIBS}        # link juga ke hidapi dan libusb
        )
    else()
        target_link_libraries(main
            ${PROJECT_NAME}
            ${OpenCV_LIBS}
            ${DEP_LIBS}        # link juga ke hidapi dan libusb
        )
    endif()
endif()
