/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#include "fruitEnv.h"



// constructor
fruitEnv::fruitEnv()
{
	agentX   = 0;
	agentY   = 0;
	agentDir = 0;
	agentVel = 0;
	
	envWidth  = 0;
	envHeight = 0;
}


// destructor
fruitEnv::~fruitEnv()
{
	
}


// Create
fruitEnv* fruitEnv::Create( int env_width, int env_height )
{
	fruitEnv* env = new fruitEnv();
	
	if( !env )
		return NULL;
	
	if( !env->init(env_width, env_height) )
		return NULL;
	
	return env;
}


// init
bool fruitEnv::init( int env_width, int env_height )
{
	envWidth  = env_width;
	envHeight = env_height;
	
	return true;
}


// Action
void fruitEnv::Action( AgentAction action )
{
	if( action == ACTION_FORWARD )
		agentVel++;
	else if( action == ACTION_BACKWARD )
		agentVel--;
	else if( action == ACTION_RIGHT )
		agentDir++;
	else if( action == ACTION_LEFT )
		agentDir--;
	
	// limit velocity
	const int maxVelocity = 1;	
	
	if( agentVel < -maxVelocity )
		agentVel = -maxVelocity;
	else if( agentVel > maxVelocity )
		agentVel = maxVelocity;
	
	// limit heading
	if( agentDir < 0 )
		agentDir += 360;
	else if( agentDir >= 360 )
		agentDir -= 360;
}
	

// Render
bool fruitEnv::Render( uint32_t width, uint32_t height )
{
	return true;
}
	

// Reset
void fruitEnv::Reset()
{
	
}