
cmake_minimum_required(VERSION 2.8)
project(jetson-reinforcement)

set(BUILD_DEPS "YES" CACHE BOOL "If YES, will install dependencies into sandbox.  Automatically reset to NO after dependencies are installed.")
set(BUILD_OPENBLAS "YES" CACHE BOOL "If YES, will download & build OpenBLAS (for Torch) into sandbox")
set(BUILD_PYTORCH "YES" CACHE BOOL "If YES, will build PyTorch and PyTorch examples")
set(BUILD_TORCH "NO" CACHE BOOL "If YES, will build LUA, Torch, and Torch examples")


find_package(CUDA)

# Pass options to NVCC
set(
	CUDA_NVCC_FLAGS
	${CUDA_NVCC_FLAGS};
	-O3 
	-gencode arch=compute_53,code=sm_53
	-gencode arch=compute_62,code=sm_62
)

#configure_file("lua/NEON.c" ${PROJECT_BINARY_DIR} COPYONLY)

# if this is the first time running cmake, perform pre-build dependency install script (or if the user manually triggers re-building the dependencies)
if( ${BUILD_DEPS} )
	message("Launching pre-build dependency installer script...")

	execute_process(COMMAND sh ../CMakePreBuild.sh ${BUILD_OPENBLAS} ${BUILD_PYTORCH} ${BUILD_TORCH}
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


# set preprocessor defines
if( ${BUILD_TORCH} )
	add_definitions(-DUSE_LUA)
	include_directories(${PROJECT_BINARY_DIR}/torch/include ${PROJECT_BINARY_DIR}/torch/include/TH ${PROJECT_INCLUDE_DIR})
	link_directories(${PROJECT_BINARY_DIR}/torch/lib)
endif()

if( ${BUILD_PYTORCH} )
	add_definitions(-DUSE_PYTHON)
	include_directories(/usr/include/python2.7)
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
if( ${BUILD_PYTORCH} )

	target_link_libraries(jetson-reinforcement python2.7)

	# transfer all python scripts to bin directory
	file(GLOB pyScripts python/*.py)

	foreach(pyScript ${pyScripts})
		message("-- Copying ${pyScript}")
		configure_file(${pyScript} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} COPYONLY)
	endforeach()
	
endif()


# LUA-specific post-build steps
if( ${BUILD_TORCH} )

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

