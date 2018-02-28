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

include_directories(${GAZEBO_INCLUDE_DIRS} gazebo-pkgs/gazebo_grasp_plugin/include)
link_directories(${GAZEBO_LIBRARY_DIRS} gazebo-pkgs/gazebo_grasp_plugin/include)

add_library(gazeboArmPlugin SHARED ArmPlugin.cpp)
add_library(gazeboPropPlugin SHARED PropPlugin.cpp)
add_library(gazeboRoverPlugin SHARED RoverPlugin.cpp)
add_library(gazeboGraspPlugin SHARED gazebo-pkgs/gazebo_grasp_plugin/src/GazeboGraspFix.cpp gazebo-pkgs/gazebo_grasp_plugin/src/GazeboGraspGripper.cpp)

target_link_libraries(gazeboArmPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)
target_link_libraries(gazeboPropPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)
target_link_libraries(gazeboRoverPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)
target_link_libraries(gazeboGraspPlugin jetson-reinforcement ${GAZEBO_LIBRARIES} ${Boost_LIBRARIES} ${PROTOBUF_LIBRARIES} pthread)

# transfer all gazebo scripts to bin directory
file(GLOB gazeboScripts *.world *.sdf *.sh)

foreach(gazeboScript ${gazeboScripts})
	message("-- Copying ${gazeboScript}")
	configure_file(${gazeboScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()

endif()

# create symbolic link for meshes
execute_process( COMMAND "${CMAKE_COMMAND}" "-E" "create_symlink" "${PROJECT_SOURCE_DIR}/gazebo/meshes" "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/meshes" )

