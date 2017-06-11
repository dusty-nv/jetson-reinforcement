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
}

#include <TH/THTensor.h>
#include <luaT.h>



bool init( const char* script_filename )
{
	lua_State* L = luaL_newstate();

	if( !L )
	{
		printf("[deepRL]  failed to create lua_State (luaL_newstate returned NULL)\n");
		return false;
	}

	printf("[deepRL]  created new lua_State\n");

	luaL_openlibs(L);
	printf("[deepRL]  opened LUA libraries\n");
	printf("[deepRL]  loading '%s'\n\n", script_filename);

 	// load and run file
	const int res = luaL_dofile(L, script_filename);

	if( res == 1 ) 
	{
		printf("Error executing resource: %s\n", script_filename);
		const char* luastr = lua_tostring(L,-1);

		if( luastr != NULL )
			printf("%s\n", luastr);
	}

	printf("\n[deepRL]  closing lua_State\n");
	lua_close(L);

	return true;
}


int main( int argc, char** argv )
{
	printf("deepRL-console (lua)\n\n");

	const char* script_filename = "test-packages.lua";
	
	if( argc > 1 )
		script_filename = argv[1];

	if( !init(script_filename) )
	{
		printf("failed to run lua script '%s', exiting deepRL-console\n", script_filename);
		return 0;
	}
	
	return 0;
}

#endif

