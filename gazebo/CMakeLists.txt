cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # -std=gnu++11

find_package(Boost REQUIRED COMPONENTS system)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include (FindPkgConfig)
if (PKG_CONFIG_FOUND)
  pkg_check_modules(GAZEBO gazebo)
endif()


if (GAZEBO_FOUND)

message("-- Gazebo7 installation detected, building plugins...")

include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(gazeboArmPlugin SHARED ArmPlugin.cpp)
add_library(gazeboPropPlugin SHARED PropPlugin.cpp)
add_library(gazeboRoverPlugin SHARED RoverPlugin.cpp)

target_link_libraries(gazeboArmPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)
target_link_libraries(gazeboPropPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)
target_link_libraries(gazeboRoverPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)


# transfer all LUA scripts to bin directory
file(GLOB gazeboScripts *.world *.sdf *.sh)

foreach(gazeboScript ${gazeboScripts})
	message("-- Copying ${gazeboScript}")
	configure_file(${gazeboScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()

endif()
