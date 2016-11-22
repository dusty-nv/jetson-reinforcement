#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
TORCH_PREFIX=$PWD/torch

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:       $BUILD_ROOT"
echo "[Pre-build]  installing Torch/LUA into:  $TORCH_PREFIX"

# (don't) break on errors
#set -e

# Install dependencies for Torch:
echo "[Pre-build]  installing Torch7 package dependencies"

sudo apt-get update
sudo apt-get install -y gfortran build-essential gcc g++ cmake curl libreadline-dev git-core liblua5.1-0-dev
# note:  gfortran is for OpenBLAS LAPACK compilation
##sudo apt-get install -qqy libgd-dev
sudo apt-get update

echo "[Pre-build]  Torch7's package dependencies have been installed"



# Install openBLAS
echo "[Pre-build]  build OpenBLAS?  $1"

if [ $1 = "ON" ] || [ $1 = "YES" ] || [ $1 = "Y" ]; then
	echo "[Pre-build]  building OpenBLAS...";
	rm -rf OpenBLAS
	git clone http://github.org/xianyi/OpenBLAS
	cd OpenBLAS
	mkdir build
	make
	make install PREFIX=$BUILD_ROOT/OpenBLAS/build
	export CMAKE_LIBRARY_PATH=$BUILD_ROOT/OpenBLAS/build/include:$BUILD_ROOT/OpenBLAS/build/lib:$CMAKE_LIBRARY_PATH
	cd $BUILD_ROOT
fi


# Install luajit-rocks into Torch dir
echo "[Pre-build]  configuring luajit-rocks"
cd $BUILD_ROOT
rm -rf luajit-rocks
git clone https://github.com/torch/luajit-rocks.git
cd luajit-rocks
mkdir -p build
cd build
git checkout master; git pull
rm -f CMakeCache.txt
cmake .. -DWITH_LUAJIT21=yes -DCMAKE_INSTALL_PREFIX=$TORCH_PREFIX -DCMAKE_BUILD_TYPE=Release
make
make install
cd $BUILD_ROOT

path_to_nvcc=$(which nvcc)
if [ -x "$path_to_nvcc" ]
then
    cutorch=ok
    cunn=ok
    echo "[Pre-build]  detected NVCC / CUDA toolkit"
fi  



sudo apt-get install gnuplot gnuplot-qt


# Install base packages:
echo "[Pre-build]  installing luarocks packages"

cd $BUILD_ROOT
git clone http://github.com/torch/rocks
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/luaffi-scm-1.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/cwrap-scm-1.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/paths-scm-1.rockspec

#$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/torch-scm-1.rockspec
echo "[Pre-build]  installing torch7 from source"
cd $BUILD_ROOT
git clone http://github.com/torch/torch7
cd torch7

# patch neon vector itrinsics (this should be fixed in master now)
# cp ../NEON.c lib/TH/vector/NEON.c
# cat lib/TH/vector/NEON.c

# patch set 993
#sed -i '6 a STRING(REGEX REPLACE "^.*(asimd).*$" "\\\\1" ASIMD_THERE ${CPUINFO})' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '7 a STRING(COMPARE EQUAL "asimd" "${ASIMD_THERE}" ASIMD_TRUE)' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '8 a IF (ASIMD_TRUE)' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '9 a set(ASIMD_FOUND true CACHE BOOL "ASIMD/NEON available on host")' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '10 a ELSE (ASIMD_TRUE)' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '11 a set(ASIMD_FOUND false CACHE BOOL "ASIMD/NEON available on host")' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
#sed -i '12 a ENDIF (ASIMD_TRUE)' $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake

#sed -i '64 a IF (ASIMD_FOUND)' $BUILD_ROOT/torch7/lib/TH/CMakeLists.txt
#sed -i '65 a MESSAGE(STATUS "asimd/Neon found with compiler flag : -D__NEON__")' $BUILD_ROOT/torch7/lib/TH/CMakeLists.txt
#sed -i '66 a SET(CMAKE_C_FLAGS "-D__NEON__ ${CMAKE_C_FLAGS}")' $BUILD_ROOT/torch7/lib/TH/CMakeLists.txt
#sed -i '67 a ENDIF (ASIMD_FOUND)' $BUILD_ROOT/torch7/lib/TH/CMakeLists.txt

cat $BUILD_ROOT/torch7/lib/TH/cmake/FindARM.cmake
cat $BUILD_ROOT/torch7/lib/TH/CMakeLists.txt

$TORCH_PREFIX/bin/luarocks make $BUILD_ROOT/torch7/rocks/torch-scm-1.rockspec
cd $BUILD_ROOT
#$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/torch7/rocks/torch-scm-1.rockspec

echo "[Pre-build]  done installing torch7 package"
echo "[Pre-build]  installing additional packages for Torch"

echo "[Pre-build]  installing nn from source"
git clone http://github.com/torch/nn
cd nn
#sed -i 's/ptrdiff_t/long/g' lib/THNN/init.c
#sed -i 's/ptrdiff_t/long/g' lib/THNN/generic/* 
$TORCH_PREFIX/bin/luarocks make $BUILD_ROOT/nn/rocks/nn-scm-1.rockspec
cd $BUILD_ROOT
echo "[Pre-build]  done installing nn package"

#$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/nn-scm-1.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/nnx-0.1-1.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/optim-1.0.5-0.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/rocks/gnuplot-scm-1.rockspec

#$TORCH_PREFIX/bin/luarocks install cwrap
#$TORCH_PREFIX/bin/luarocks install paths
#$TORCH_PREFIX/bin/luarocks install torch
#$TORCH_PREFIX/bin/luarocks install nn
#$TORCH_PREFIX/bin/luarocks install nnx
#$TORCH_PREFIX/bin/luarocks install optim
#$TORCH_PREFIX/bin/luarocks install cutorch
#$TORCH_PREFIX/bin/luarocks install trepl
#$TORCH_PREFIX/bin/luarocks install gnuplot

echo "[Pre-build]  installing cutorch from source"
git clone https://github.com/torch/cutorch
sed -i 's/$(getconf _NPROCESSORS_ONLN)/1/g' cutorch/rocks/cutorch-1.0-0.rockspec
sed -i 's/$(getconf _NPROCESSORS_ONLN)/1/g' cutorch/rocks/cutorch-scm-1.rockspec
sed -i 's/jopts=3/jopts=1/g' cutorch/rocks/cutorch-1.0-0.rockspec
sed -i 's/jopts=3/jopts=1/g' cutorch/rocks/cutorch-scm-1.rockspec

$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/cutorch/rocks/cutorch-scm-1.rockspec
echo "[Pre-build]  done installing cutorch package"

echo "[Pre-build]  installing cudnn bindings from source"
# install cudnn v5 bindings
#git clone -b R5 http://github.com/soumith/cudnn.torch 
git clone http://github.com/soumith/cudnn.torch 
#sed -i 's/ffi.sizeof('half'),/2,/g' cudnn.torch/init.lua
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/cudnn.torch/cudnn-scm-1.rockspec
echo "[Pre-build]  done installing cudnn bindings package"

echo ""
echo "[Pre-build]  Torch7 has been installed successfully"
echo ""

#echo "installing iTorch"
#sudo apt-get install libzmq3-dev libssl-dev python-zmq
#sudo pip install ipython
#ipython --version
## pip uninstall IPython
## pip install ipython==3.2.1
#sudo pip install jupyter
#git clone https://github.com/facebook/iTorch.git
#$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/iTorch/itorch-scm-1.rockspec


