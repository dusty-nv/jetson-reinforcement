
cmake_minimum_required(VERSION 2.8)
project(jetson-reinforcement)

set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_OPENBLAS "YES" CACHE BOOL "If YES, will download & build OpenBLAS (for Torch) into sandbox")

find_package(CUDA)

# Pass options to NVCC
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3 -gencode arch=compute_53,code=sm_53
)

configure_file("lua/NEON.c" ${PROJECT_BINARY_DIR} COPYONLY)

# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_OPENBLAS}
				WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
				RESULT_VARIABLE PREBUILD_SCRIPT_RESULT)

	set(BUILD_DEPS "NO" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed." FORCE)
	message("Finished installing dependencies")
endif()





# setup project output paths
set(PROJECT_OUTPUT_DIR  ${PROJECT_BINARY_DIR}/${CMAKE_SYSTEM_PROCESSOR})
set(PROJECT_INCLUDE_DIR ${PROJECT_OUTPUT_DIR}/include)

file(MAKE_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
file(MAKE_DIRECTORY ${PROJECT_INCLUDE_DIR})

message("-- system arch:  ${CMAKE_SYSTEM_PROCESSOR}")
message("-- output path:  ${PROJECT_OUTPUT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# build deepRL C/C++ interface
include_directories(${PROJECT_BINARY_DIR}/torch/include ${PROJECT_BINARY_DIR}/torch/include/TH ${PROJECT_INCLUDE_DIR})
link_directories(${PROJECT_BINARY_DIR}/torch/lib)

file(GLOB deepRLSources c/*.cpp cuda/*.cu)
file(GLOB deepRLIncludes c/*.h cuda/*.h)


cuda_add_library(jetson-reinforcement SHARED ${deepRLSources})
target_link_libraries(jetson-reinforcement luajit luaT TH THC)


# transfer all headers to the include directory
foreach(include ${deepRLIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()

# transfer all LUA scripts to bin directory
file(GLOB luaScripts lua/*.lua lua/*.ipynb)

foreach(luaScript ${luaScripts})
	message("-- Copying ${luaScript}")
	configure_file(${luaScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
endforeach()


# build samples & tools
add_subdirectory(gazebo)
add_subdirectory(samples)
add_subdirectory(tools)
