cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17) # Ensure all components use C++17
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)
project(stereo_vio)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g")  # -g for debugging

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall   -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
find_package(OpenCV 4.2.0)
if(NOT OpenCV_FOUND)
  find_package(OpenCV 3.2)
  if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV > 3.0 not found.")
  endif()
endif()

MESSAGE("OPENCV VERSION:")
MESSAGE(${OpenCV_VERSION})

find_package(PCL REQUIRED)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Iridescence REQUIRED)
find_package(GTSAM REQUIRED)

add_library(${PROJECT_NAME} SHARED
src/System.cpp
src/Settings.cpp
src/Camera.cpp
src/Visualization.cpp
src/Map.cpp
src/FeatureExtractor.cpp
src/FeatureTracker.cpp
src/FeatureMatcher.cpp
src/KeyFrame.cpp
src/OptimizationBA.cpp
include/OptimizationBA.h
include/KeyFrame.h
include/FeatureMatcher.h
include/FeatureTracker.h
include/FeatureExtractor.h
include/Map.h
include/Visualization.h
include/Camera.h
include/System.h
include/Settings.h
include/Conversions.h
)

include_directories(
  ${PCL_INCLUDE_DIRS} 
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIR}
  ${YAML_CPP_INCLUDE_DIR}
  )

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${YAML_CPP_LIBRARIES}
Iridescence::Iridescence
gtsam
)

add_executable(VIOSlam src/VIOSlam.cpp)
target_link_libraries(VIOSlam ${PROJECT_NAME})

add_executable(VIOSlamMono src/VIOSlamMono.cpp)
target_link_libraries(VIOSlamMono ${PROJECT_NAME})