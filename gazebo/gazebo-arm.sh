#!/bin/sh

echo " "
echo "configuring gazebo7 plugin paths"
echo "previous GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "script directory $SCRIPT_DIR"

MY_PLUGIN_PATH=$SCRIPT_DIR/../lib
echo "plugin path $MY_PLUGIN_PATH"

export GAZEBO_PLUGIN_PATH=$MY_PLUGIN_PATH:$GAZEBO_PLUGIN_PATH
echo "new GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"

echo " "
echo "starting gazebo7 simulator"
gazebo gazebo-arm.world --verbose
