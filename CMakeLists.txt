
cmake_minimum_required(VERSION 2.8)
project(jetson-reinforcement)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11

set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_OPENBLAS "YES" CACHE BOOL "If YES, will download & build OpenBLAS (for Torch) into sandbox")
set(USE_PYTHON "YES" CACHE BOOL "If YES, will build PyTorch and PyTorch examples")
set(USE_LUA "NO" CACHE BOOL "If YES, will build LUA, Torch, and Torch examples")

	
# configure CUDA
find_package(CUDA)

set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS}; 
    -O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_60,code=sm_60
	-gencode arch=compute_61,code=sm_61
	-gencode arch=compute_62,code=sm_62
)


# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_OPENBLAS} ${USE_PYTHON} ${USE_LUA}
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
message("-- nvcuda path:  ${CUDA_TOOLKIT_ROOT_DIR}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_OUTPUT_DIR}/lib)


# set preprocessor defines
if( ${USE_LUA} )
	add_definitions(-DUSE_LUA)
	include_directories(${PROJECT_BINARY_DIR}/torch/include ${PROJECT_BINARY_DIR}/torch/include/TH ${PROJECT_INCLUDE_DIR})
	link_directories(${PROJECT_BINARY_DIR}/torch/lib)
endif()

if( ${USE_PYTHON} )
	add_definitions(-DUSE_PYTHON)
	include_directories(/usr/include/python2.7 ${PROJECT_BINARY_DIR}/pytorch/torch/lib/include ${PROJECT_BINARY_DIR}/pytorch/torch/csrc ${PROJECT_BINARY_DIR}/pytorch/torch/csrc/cuda ${PROJECT_BINARY_DIR}/pytorch ${PROJECT_BINARY_DIR}/pytorch/torch/lib/include/TH ${PROJECT_INCLUDE_DIR})
	link_directories(${PROJECT_BINARY_DIR}/pytorch/torch/lib ${PROJECT_BINARY_DIR}/pytorch/torch)
	link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()




# library sources
file(GLOB deepRLSources c/*.cpp cuda/*.cu)
file(GLOB deepRLIncludes c/*.h cuda/*.h)

cuda_add_library(jetson-reinforcement SHARED ${deepRLSources})


# transfer all headers to the include directory
foreach(include ${deepRLIncludes})
	message("-- Copying ${include}")
	configure_file(${include} ${PROJECT_INCLUDE_DIR} COPYONLY)
endforeach()


# Python post-build steps
if( ${USE_PYTHON} )
	target_link_libraries(jetson-reinforcement python2.7 _C ATen cuda nvrtc)

	# transfer all python scripts to bin directory
	file(GLOB pyScripts python/*.py)

	foreach(pyScript ${pyScripts})
		message("-- Copying ${pyScript}")
		configure_file(${pyScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
	endforeach()
endif()


# LUA post-build steps
if( ${USE_LUA} )
	target_link_libraries(jetson-reinforcement luajit luaT TH THC)

	# transfer all LUA scripts to bin directory
	file(GLOB luaScripts lua/*.lua lua/*.ipynb)

	foreach(luaScript ${luaScripts})
		message("-- Copying ${luaScript}")
		configure_file(${luaScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
	endforeach()

	file(COPY lua/DDPG DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()


add_subdirectory(gazebo)
add_subdirectory(samples)
add_subdirectory(tools)
add_subdirectory(utils)

target_link_libraries(jetson-reinforcement jetson-utils)
