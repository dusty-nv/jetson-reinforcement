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

#ifndef __DEV_KEYBOARD_H__
#define __DEV_KEYBOARD_H__

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>

#include <linux/input-event-codes.h>


/**
 * Keyboard device
 */
class KeyboardDevice
{
public:
	/**
	 * Create device
	 */
	static KeyboardDevice* Create( const char* path="/dev/input/by-path/platform-i8042-serio-0-event-kbd" );

	/**
	 * Destructor
	 */
	~KeyboardDevice();

	/**
	 * Poll the device for updates
	 */
	bool Poll( uint32_t timeout=0 );

	/**
	 * Check if a particular key is pressed
	 */
	bool KeyDown( uint32_t code ) const;

	/**
	 * Enable/disable verbose logging
	 */
	void Debug( bool enabled=true );

protected:
	// constructor
	KeyboardDevice();

	static const int MAX_KEYS = 256;

	int  mKeyMap[MAX_KEYS];
	int  mFD;
	bool mDebug;

	std::string mPath;
};

#endif

