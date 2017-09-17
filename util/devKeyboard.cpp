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

#include "devKeyboard.h"

#include <linux/input.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>


// constructor
KeyboardDevice::KeyboardDevice()
{
	mFD    = -1;
	mDebug = false;

	memset(mKeyMap, 0, sizeof(mKeyMap));
}


// destructor
KeyboardDevice::~KeyboardDevice()
{

}


// Create
KeyboardDevice* KeyboardDevice::Create( const char* path )
{
	if( !path )
		return NULL;

	const int fd = open(path, O_RDONLY);

	if( fd == -1 )
	{
		printf("keyboard -- failed to open %s\n", path);
		return NULL;
	}

	KeyboardDevice* kbd = new KeyboardDevice();

	kbd->mFD   = fd;
	kbd->mPath = path;

	return kbd;
}


// Poll
bool KeyboardDevice::Poll( uint32_t timeout )
{
	const uint32_t max_ev = 64;
	struct input_event ev[max_ev];

	fd_set fds;
	FD_ZERO(&fds);
	FD_SET(mFD, &fds);

	struct timeval tv;
 
	tv.tv_sec  = 0;
	tv.tv_usec = timeout*1000;

	const int result = select(mFD + 1, &fds, NULL, NULL, &tv);

	if( result == -1 ) 
	{
		printf("keyboard -- select() failed (errno=%i) (%s)\n", errno, strerror(errno));
		return false;
	}
	else if( result == 0 )
	{
		if( mDebug && timeout > 0 )
			printf("keyboard -- select() timed out...\n");

		return false;	// timeout, not necessarily an error (TRY_AGAIN)
	}

	const int bytesRead = read(mFD, ev, sizeof(struct input_event) * max_ev);

	if( bytesRead < (int)sizeof(struct input_event) ) 
	{
		printf("keyboard -- read() expected %d bytes, got %d\n", (int)sizeof(struct input_event), bytesRead);
		return false;
	}

	const int num_ev = bytesRead / sizeof(struct input_event);

	for( int i = 0; i < num_ev; i++ ) 
	{
		if( ev[i].type != EV_KEY )
			continue;

		if( ev[i].value < 0 || ev[i].value > 2 )
			continue;

		if( ev[i].code >= MAX_KEYS )
			continue;

		mKeyMap[ev[i].code] = (ev[i].value == 0) ? false : true;

		if( mDebug )
			printf("keyboard -- code %02u  value %i\n", ev[i].code, ev[i].value);
	}

	return true;	
}


// KeyDown
bool KeyboardDevice::KeyDown( uint32_t code ) const
{
	if( code >= MAX_KEYS )
		return false;

	return mKeyMap[code];
}


// Debug
void KeyboardDevice::Debug( bool enable )
{
	mDebug = enable;
}

