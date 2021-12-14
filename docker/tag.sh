#!/usr/bin/env bash

# find OS version
ARCH=$(uname -i)
echo "ARCH:  $ARCH"

if [ $ARCH = "aarch64" ]; then
	L4T_VERSION_STRING=$(head -n 1 /etc/nv_tegra_release)

	L4T_RELEASE=$(echo $L4T_VERSION_STRING | cut -f 2 -d ' ' | grep -Po '(?<=R)[^;]+')
	L4T_REVISION=$(echo $L4T_VERSION_STRING | cut -f 2 -d ',' | grep -Po '(?<=REVISION: )[^;]+')

	L4T_REVISION_MAJOR=${L4T_REVISION:0:1}
	L4T_REVISION_MINOR=${L4T_REVISION:2:1}

	L4T_VERSION="$L4T_RELEASE.$L4T_REVISION"

	echo "L4T BSP Version:  L4T R$L4T_VERSION"
fi

if [ $ARCH = "aarch64" ]; then
	TAG="r$L4T_VERSION"
	
	if [ $L4T_VERSION = "32.5.1" ] || [ $L4T_VERSION = "32.5.2" ]; then
		TAG="r32.5.0"
	fi	
elif [ $ARCH = "x86_64" ]; then
	TAG="$ARCH"
else
	echo "unsupported architecture:  $ARCH"
	exit 1
fi

CONTAINER_NAME="jetson-reinforcement"


