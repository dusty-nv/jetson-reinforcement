#!/usr/bin/env bash
#
# Start an instance of the jetson-reinforcement docker container.
# See below or run this script with -h or --help to see usage options.
#
# This script should be run from the root dir of the jetson-reinforcement project:
#
#     $ cd /path/to/your/jetson-reinforcement
#     $ docker/run.sh
#

show_help() {
    echo " "
    echo "usage: Starts the Docker container and runs a user-specified command"
    echo " "
    echo "   ./docker/run.sh --container DOCKER_IMAGE"
    echo "                   --volume HOST_DIR:MOUNT_DIR"
    echo "                   --run RUN_COMMAND"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   -c, --container DOCKER_IMAGE Specifies the name of the Docker container"
    echo "                                image to use (default: 'jetson-reinforcement')"
    echo " "
    echo "   --ros ROS_DISTRO Starts the version of the container using the"
    echo "                    specified ROS distro (or foxy if not specified)"
    echo "                    This is overridden by the --container argument"
    echo " "
    echo "   -d, --dev  Runs the container in development mode, where the source"
    echo "              files are mounted into the container dynamically, so they"
    echo "              can more easily be edited from the host machine."
    echo " "
    echo "   -v, --volume HOST_DIR:MOUNT_DIR Mount a path from the host system into"
    echo "                                   the container.  Should be specified as:"
    echo " "
    echo "                                      -v /my/host/path:/my/container/path"
    echo " "
    echo "                                   (these should be absolute paths)"
    echo " "
    echo "   -r, --run RUN_COMMAND  Command to run once the container is started."
    echo "                          Note that this argument must be invoked last,"
    echo "                          as all further arguments will form the command."
    echo "                          If no run command is specified, an interactive"
    echo "                          terminal into the container will be provided."
    echo " "
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

# find container tag from os version
source docker/tag.sh

# where the project resides inside docker
DOCKER_ROOT="/jetson-reinforcement"	

# generate mount commands
DATA_VOLUME="--volume $PWD/data:$DOCKER_ROOT/data"
DEV_VOLUME=""

# parse user arguments
USER_VOLUME=""
USER_COMMAND=""

while :; do
    case $1 in
        -h|-\?|--help)
            show_help    # Display a usage synopsis.
            exit
            ;;
        -c|--container)       # Takes an option argument; ensure it has been specified.
            if [ "$2" ]; then
                CONTAINER_IMAGE=$2
                shift
            else
                die 'ERROR: "--container" requires a non-empty option argument.'
            fi
            ;;
        --container=?*)
            CONTAINER_IMAGE=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --container=)         # Handle the case of an empty --image=
            die 'ERROR: "--container" requires a non-empty option argument.'
            ;;
	   --ros)
            if [ "$2" ]; then
                ROS_DISTRO=$2
                shift
            else
                ROS_DISTRO="foxy"
            fi
            ;;
        --ros=?*)
            ROS_DISTRO=${1#*=} # Delete everything up to "=" and assign the remainder.
            ;;
        --ros=)         # Handle the case of an empty --image=
            ROS_DISTRO="foxy"
            ;;
	   -d|--dev)
            DEV_VOLUME="--volume $PWD/jetson_reinforcement:$DOCKER_ROOT/jetson_reinforcement --volume $PWD/examples:$DOCKER_ROOT/examples --volume $PWD/scripts:$DOCKER_ROOT/scripts --volume $PWD/tests:$DOCKER_ROOT/tests"
            ;;
        -v|--volume)
            if [ "$2" ]; then
                USER_VOLUME=" -v $2 "
                shift
            else
                die 'ERROR: "--volume" requires a non-empty option argument.'
            fi
            ;;
        --volume=?*)
            USER_VOLUME=" -v ${1#*=} " # Delete everything up to "=" and assign the remainder.
            ;;
        --volume=)         # Handle the case of an empty --image=
            die 'ERROR: "--volume" requires a non-empty option argument.'
            ;;
        -r|--run)
            if [ "$2" ]; then
                shift
                USER_COMMAND=" $@ "
            else
                die 'ERROR: "--run" requires a non-empty option argument.'
            fi
            ;;
        --)              # End of all options.
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)               # Default case: No more options, so break out of the loop.
            break
    esac

    shift
done

# select the container, unless --container was explicitly specified
if [ -z "$CONTAINER_IMAGE" ]; then
	CONTAINER_IMAGE="$CONTAINER_NAME:$TAG"

	if [ -n "$ROS_DISTRO" ]; then
		CONTAINER_IMAGE="$CONTAINER_NAME:$TAG-ros-$ROS_DISTRO"
	fi

	CONTAINER_REMOTE_IMAGE="dustynv/$CONTAINER_IMAGE"

	# check for local image
	if [[ "$(sudo docker images -q $CONTAINER_IMAGE 2> /dev/null)" == "" ]]; then
		CONTAINER_IMAGE=$CONTAINER_REMOTE_IMAGE
	fi
fi

echo "CONTAINER:     $CONTAINER_IMAGE"
echo "DEV_VOLUME:    $DEV_VOLUME"
echo "DATA_VOLUME:   $DATA_VOLUME"
echo "USER_VOLUME:   $USER_VOLUME"
echo "USER_COMMAND:  $USER_COMMAND"

MOUNTS="\
--volume /etc/timezone:/etc/timezone:ro \
--volume /etc/localtime:/etc/localtime:ro \
$DEV_VOLUME \
$DATA_VOLUME \
$USER_VOLUME"

if [ $ARCH = "aarch64" ]; then

	sudo docker run --runtime nvidia -it --rm \
		--name=$CONTAINER_NAME \
		--network host \
		$MOUNTS $CONTAINER_IMAGE $USER_COMMAND
	    
elif [ $ARCH = "x86_64" ]; then

	sudo docker run --gpus all -it --rm \
		--name=$CONTAINER_NAME \
		--network=host \
		--shm-size=8g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		$MOUNTS $CONTAINER_IMAGE $USER_COMMAND
		
fi
