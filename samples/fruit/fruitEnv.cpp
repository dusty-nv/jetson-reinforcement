/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#include "FruitEnv.h"



// constructor
FruitEnv::FruitEnv()
{
	agentX   = 0;
	agentY   = 0;
	agentDir = 0;
	agentVel = 0;
	
	worldWidth   = 0;
	worldHeight  = 0;
	renderWidth  = 0;
	renderHeight = 0;
	
	imageCPU = NULL;
	imageGPU = NULL;
}


// destructor
FruitEnv::~FruitEnv()
{
	
}


// Create
FruitEnv* FruitEnv::Create( uint32_t world_width, uint32_t world_height, uint32_t render_width, uint32_t render_height )
{
	FruitEnv* env = new FruitEnv();
	
	if( !env )
		return NULL;
	
	if( !env->init(world_width, world_height, render_width, render_height) )
		return NULL;
	
	return env;
}


// init
bool FruitEnv::init( uint32_t world_width, uint32_t world_height, uint32_t render_width, uint32_t render_height )
{
	worldWidth   = world_width;
	worldHeight  = world_height;
	renderWidth  = render_width;
	renderHeight = render_height;
	
	return true;
}


// Action
void FruitEnv::Action( AgentAction action )
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
float* FruitEnv::Render()
{
	return true;
}
	

// Reset
void FruitEnv::Reset()
{
	
}