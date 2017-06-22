/*
 * deepRL
 */

#include <stdio.h>


#ifdef USE_LUA

extern "C" 
{ 
#include <lua.h>
#include <lauxlib.h>
#include <lualib.h>
//#include <TH/THTensor.h>
}

#include <TH/THTensor.h>
#include <luaT.h>


#define SCRIPT_FILENAME "test-packages.lua"


bool init()
{
	lua_State* L = luaL_newstate();

	if( !L )
	{
		printf("[deepRL]  failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("[deepRL]  created new lua_State\n");

	luaL_openlibs(L);
	printf("[deepRL]  opened lua libraries\n");

 	// load and run file
	const int res = luaL_dofile(L, SCRIPT_FILENAME);

	if( res == 1 ) 
	{
		printf("Error executing resource: %s\n", SCRIPT_FILENAME);
		const char* luastr = lua_tostring(L,-1);

		if( luastr != NULL )
			printf("%s\n", luastr);
	}

	printf("[deepRL]  closing lua_State\n");
	lua_close(L);

	return true;
}


int main( int argc, char** argv )
{
	printf("deepRL-diagnostic (lua)\n\n");
	
	if( !init() )
	{
		printf("failed to init lua, exiting deepRL-console\n");
		return 0;
	}
	
	return 0;
}

#endif

