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

#ifndef __DEV_JOYSTICK_H__
#define __DEV_JOYSTICK_H__

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <string>


/**
 * Joystick device
 */
class JoystickDevice
{
public:
	/**
	 * Create device
	 */
	static JoystickDevice* Create( const char* device="Microntek              USB Joystick          " );

	/**
	 * Destructor
	 */
	~JoystickDevice();

	/**
	 * Poll the device for updates
	 */
	bool Poll( uint32_t timeout=0 );

	/**
	 * Enable/disable verbose logging
	 */
	void Debug( bool enabled=true );

protected:
	// constructor
	JoystickDevice();

	static const int MAX_AXIS = 256;

	float mAxisNorm[MAX_AXIS];
	int   mAxisRaw[MAX_AXIS];	
	int   mFD;
	bool  mDebug;

	std::string mPath;
};

#endif
