#!/usr/bin/env bash
# this script is automatically run from CMakeLists.txt

BUILD_ROOT=$PWD
TORCH_PREFIX=$PWD/torch

echo "[Pre-build]  dependency installer script running..."
echo "[Pre-build]  build root directory:       $BUILD_ROOT"
echo "[Pre-build]  installing Torch/LUA into:  $TORCH_PREFIX"


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

$TORCH_PREFIX/bin/luarocks install cwrap
$TORCH_PREFIX/bin/luarocks install classic
$TORCH_PREFIX/bin/luarocks install paths
$TORCH_PREFIX/bin/luarocks install torch
$TORCH_PREFIX/bin/luarocks install nn
$TORCH_PREFIX/bin/luarocks install nnx
$TORCH_PREFIX/bin/luarocks install optim
#$TORCH_PREFIX/bin/luarocks install cutorch
$TORCH_PREFIX/bin/luarocks install trepl
$TORCH_PREFIX/bin/luarocks install gnuplot

git clone http://github.com/soumith/cutorch
sed -i 's/-j$(getconf _NPROCESSORS_ONLN)/-j1/g' cutorch/rocks/cutorch-1.0-0.rockspec
sed -i 's/-j$(getconf _NPROCESSORS_ONLN)/-j1/g' cutorch/rocks/cutorch-scm-1.rockspec
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/cutorch/rocks/cutorch-scm-1.rockspec

# install cudnn v5 bindings
git clone -b R5 http://github.com/soumith/cudnn.torch 
$TORCH_PREFIX/bin/luarocks install $BUILD_ROOT/cudnn.torch/cudnn-scm-1.rockspec


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


