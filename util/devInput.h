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

#ifndef __DEV_INPUT_H__
#define __DEV_INPUT_H__

#include "devKeyboard.h"
#include "devJoystick.h"

#include <utility>
#include <vector>


/**
 * Typedef of device <path, name> pairs
 */
typedef std::vector< std::pair<std::string, std::string> > DeviceList;


/**
 * Input device manager
 */
class InputDevices
{
public:
	/**
	 * Create device
	 */
	static InputDevices* Create();

	/**
	 * Destructor
	 */
	~InputDevices();

	/**
	 * Poll the devices for updates
	 */
	bool Poll( uint32_t timeout=0 );

	/**
 	 * Retrieve the keyboard device
	 */
	inline KeyboardDevice* GetKeyboard() const			{ return mKeyboard; }

	/**
 	 * Retrieve the gamepad device
	 */
	inline JoystickDevice* GetJoystick() const			{ return mJoystick; }

	/**
	 * Scan /dev/input for devices
	 */
	static void Enumerate( DeviceList& devices );

	/**
 	 * Find /dev/input path by device name
	 */
	static std::string FindPathByName( const char* name );

	/**
	 * Enable/disable verbose logging
	 */
	void Debug( bool enabled=true );

protected:
	// constructor
	InputDevices();

	KeyboardDevice* mKeyboard;
	JoystickDevice* mJoystick;

	bool mDebug;
};

#endif

