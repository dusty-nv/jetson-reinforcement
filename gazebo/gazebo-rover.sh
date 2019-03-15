#!/bin/sh

echo " "
echo "configuring Gazebo7 plugin paths"
echo "previous GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "script directory $SCRIPT_DIR"

MY_PLUGIN_PATH=$SCRIPT_DIR/../lib
echo "plugin path $MY_PLUGIN_PATH"

export GAZEBO_PLUGIN_PATH=$MY_PLUGIN_PATH:$GAZEBO_PLUGIN_PATH
echo "GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH"

export GAZEBO_MODEL_PATH=$SCRIPT_DIR:$GAZEBO_MODEL_PATH
echo "GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH"

echo " "

echo "starting Gazebo7 Client (gzclient)"
gnome-terminal -e 'sh -c "echo \"\033]0; Gazebo7 Client (gzclient)\007\"; \
				echo \"launching Gazebo7 Client (gzclient)\"; \
				echo \"Press Ctrl+Q or close window to quit\n\"; \
				sleep 10; \
				gzclient --verbose; \
				pkill gzserver"' # pkill -INT gzserver

echo "starting Gazebo7 Server (gzserver)\n"
gzserver gazebo-rover.world --verbose

echo "Gazebo7 Server (gzserver) has exited."

#gazebo gazebo-rover.world --verbose
