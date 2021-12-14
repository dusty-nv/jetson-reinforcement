#!/usr/bin/env bash

ROS_DISTRO=${1:-"none"}
BASE_IMAGE=$2

# find container tag from os version
source docker/tag.sh

if [ $ARCH = "aarch64" ]; then
	if [ -z $BASE_IMAGE ]; then
		if [ $L4T_VERSION = "32.6.1" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.6.1-py3"
		elif [ $L4T_VERSION = "32.5.0" ] || [ $L4T_VERSION = "32.5.1" ]; then
			BASE_IMAGE="dustynv/l4t-ml:r32.5.0-py3"  # use alt version of l4t-ml with PyTorch 1.9
		elif [ $L4T_VERSION = "32.4.4" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.4-py3"
		elif [ $L4T_VERSION = "32.4.3" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.3-py3"
		elif [ $L4T_VERSION = "32.4.2" ]; then
			BASE_IMAGE="nvcr.io/nvidia/l4t-ml:r32.4.2-py3"
		else
			echo "cannot build jetson-reinforcement docker container for L4T R$L4T_VERSION"
			echo "please upgrade to the latest JetPack, or build jetson-reinforcement natively"
			exit 1
		fi
	fi
elif [ $ARCH = "x86_64" ]; then
	echo "building jetson-reinforcement container is not currently supported on x86"
	exit 1
fi

RL_CONTAINER="$CONTAINER_NAME:$TAG"

# build the base container
echo "CONTAINER=$RL_CONTAINER_BASE"
echo "BASE_IMAGE=$BASE_IMAGE"

sudo docker build -t $RL_CONTAINER -f Dockerfile.base \
          --build-arg BASE_IMAGE=$BASE_IMAGE \
		.
