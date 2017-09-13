/*
 * deepRL
 */

#include "devInput.h"


int main( int argc, char** argv )
{
	printf("deepRL-input\n");

	// scan for devices
	DeviceList devices;
	InputDevices::Enumerate(devices);

	// create input manager device
	InputDevices* mgr = InputDevices::Create();

	if( !mgr )
		return 0;

	// enable verbose debug text
	mgr->Debug();

	// poll for updates
	while(true)
	{
		mgr->Poll(0);
	}

	return 0;
}

