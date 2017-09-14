/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "devInput.h"

#include <stdio.h>
#include <dirent.h>
#include <fcntl.h>
#include <unistd.h>

#include <linux/input.h>
#include <sys/ioctl.h>


// constructor
InputDevices::InputDevices()
{
	mKeyboard = NULL;
	mJoystick = NULL;
	mDebug    = false;
}


// destructor
InputDevices::~InputDevices()
{

}


// Create
InputDevices* InputDevices::Create()
{
	InputDevices* mgr = new InputDevices();

	mgr->mKeyboard = KeyboardDevice::Create();

	if( !mgr->mKeyboard )
		return NULL;

	mgr->mJoystick = JoystickDevice::Create();

	return mgr;
}


// Poll
bool InputDevices::Poll( uint32_t timeout )
{
	if( !mKeyboard && !mJoystick )
		return false;

	if( mKeyboard != NULL )
		mKeyboard->Poll(timeout);
	
	if( mJoystick != NULL )
		mJoystick->Poll(timeout);

	return true;	
}


// Path used to look for input devices
#define DEV_PATH "/dev/input"
//#define DEV_PATH "/dev/input/by-path"


// Filter for the AutoDevProbe scandir on /dev/input.
// @param dir The current directory entry provided by scandir.
// @return Non-zero if the given directory entry starts with "event", or zero otherwise.
static int is_event_device(const struct dirent *dir) 
{
	return strncmp("event", dir->d_name, 5) == 0;
	//if( strcmp(dir->d_name, ".") == 0 || strcmp(dir->d_name, "..") == 0 )
	//	return 0;

	//return 1;
}


// Enumerate
void InputDevices::Enumerate( DeviceList& devices )
{
	struct dirent **namelist;
	int ndev = scandir(DEV_PATH, &namelist, is_event_device, versionsort);

	if (ndev <= 0)
	{	
		return;
	}

	printf("Available devices (%i):\n", ndev);

	for(int i = 0; i < ndev; i++)
	{
		char fname[64];
		char name[256] = "???";

		snprintf(fname, sizeof(fname), "%s/%s", DEV_PATH, namelist[i]->d_name);
		int fd = open(fname, O_RDONLY);

		if (fd < 0)
			continue;

		if( ioctl(fd, EVIOCGNAME(sizeof(name)), name) < 0 )
			continue;

		printf("%s:	'%s'\n", fname, name);

		close(fd);
		free(namelist[i]);

		devices.push_back(std::pair<std::string, std::string>(fname, name));
	}
}


// FindPathByName
std::string InputDevices::FindPathByName( const char* name )
{
	if( !name )
		return "";

	DeviceList list;
	Enumerate(list);

	const size_t numDevices = list.size();

	if( numDevices == 0 )
		return "";

	for( size_t n=0; n < numDevices; n++ )
	{
		if( strcasecmp(name, list[n].second.c_str()) == 0 )
			return list[n].first;
	}

	return "";
}


// Debug
void InputDevices::Debug( bool enable )
{
	mDebug = enable;

	if( mKeyboard != NULL )
		mKeyboard->Debug(enable);

	if( mJoystick != NULL )
		mJoystick->Debug(enable);
}



