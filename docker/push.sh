#!/usr/bin/env bash

ROS_DISTRO=${1:-"foxy"}
source docker/tag.sh

# push image
push() 
{
	local remote_image="dustynv/$1"
	
	sudo docker rmi $remote_image
	sudo docker tag $1 $remote_image
	
	echo "pushing image $remote_image"
	sudo docker push $remote_image
	echo "done pushing image $remote_image"
}

push "$CONTAINER_NAME:$TAG"

ROS_CONTAINER="$CONTAINER_NAME:$TAG-ros-$ROS_DISTRO"
push "$ROS_CONTAINER"