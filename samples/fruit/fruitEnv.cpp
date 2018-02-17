/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "fruitEnv.h"
#include "cudaMappedMemory.h"

#include <math.h>
#include "rand.h"
#include "pi.h"



// ActionToStr
const char* FruitEnv::ActionToStr( AgentAction action )
{
	if( action == ACTION_FORWARD )		return "FORWARD ";
	else if( action == ACTION_BACKWARD )	return "BACKWARD";
	else if( action == ACTION_LEFT )		return "LEFT    ";
	else if( action == ACTION_RIGHT )		return "RIGHT   ";
	else if( action == ACTION_NONE )		return "NONE    ";

	return "unknown ";
}


// constructor
FruitEnv::FruitEnv()
{
	agentX   = 0;
	agentY   = 0;
	agentDir = 0;
	agentVel = 0;
	agentRad = DEFAULT_RAD;
	
	agentColor[0] = 1.0f; 
	agentColor[1] = 0.0f; 
	agentColor[2] = 1.0f; 
	agentColor[3] = 1.0f;
	
	bgColor[0] = 1.0f; 
	bgColor[1] = 1.0f; 
	bgColor[2] = 1.0f; 
	bgColor[3] = 1.0f;
	
	epMaxFrames  = 0;
	epFrameCount = 0;
	worldWidth   = 0;
	worldHeight  = 0;
	renderWidth  = 0;
	renderHeight = 0;
	
	renderCPU = NULL;
	renderGPU = NULL;
	
	srand_time();	// seed rng
}


// destructor
FruitEnv::~FruitEnv()
{
	
}


// Create
FruitEnv* FruitEnv::Create( uint32_t world_width, uint32_t world_height, uint32_t episode_max_length )
{
	return Create(world_width, world_height, world_width, world_height, episode_max_length);
}


// Create
FruitEnv* FruitEnv::Create( uint32_t world_width, uint32_t world_height, uint32_t render_width, uint32_t render_height, uint32_t episode_max_length )
{
	FruitEnv* env = new FruitEnv();
	
	if( !env )
		return NULL;
	
	if( !env->init(world_width, world_height, render_width, render_height, episode_max_length) )
		return NULL;
	
	return env;
}


// init
bool FruitEnv::init( uint32_t world_width, uint32_t world_height, uint32_t render_width, uint32_t render_height, uint32_t episode_max_length )
{
	worldWidth   = world_width;
	worldHeight  = world_height;
	renderWidth  = render_width;
	renderHeight = render_height;
	epMaxFrames  = episode_max_length;
	
	// allocate render image memory
	if( !cudaAllocMapped((void**)&renderCPU, (void**)&renderGPU, renderWidth * renderHeight * sizeof(float) * 4) )
		return false;
	
	// allocate fruit objects
	for( uint32_t n=0; n < MAX_OBJECTS; n++ )
	{
		fruitObject* obj = new fruitObject();
		fruitObjects.push_back(obj);
	}
	
	// reset the first time
	Reset();
	return true;
}


// Action
bool FruitEnv::Action( AgentAction action, float* reward )
{	
	// first, make sure the action is valid
	if( action >= NUM_ACTIONS )
	{
		printf("FruitEnv::Action() -- invalid action selected (%i)\n", (int)action);
		return false;
	}
	
	// apply action
	const float delta = 1.0f;
		
	if( action == ACTION_FORWARD )
		agentVel += delta;
	else if( action == ACTION_BACKWARD )
		agentVel -= delta;
	else if( action == ACTION_RIGHT )
		agentDir += delta;
	else if( action == ACTION_LEFT )
		agentDir -= delta;
	
	// limit velocity
	const float maxVelocity = 1.0f;	
	
	if( agentVel < -maxVelocity )
		agentVel = -maxVelocity;
	else if( agentVel > maxVelocity )
		agentVel = maxVelocity;
	
	// limit heading
	if( agentDir < 0.0f )
		agentDir += 360.0f;
	else if( agentDir >= 360.0f )
		agentDir -= 360.0f;
	
	
	// update location
	const float rad = DEG_TO_RAD * agentDir;
	const float vx  = agentVel * cosf(rad);
	const float vy  = agentVel * sinf(rad);
	
	agentX += vx;
	agentY += vy;
	
	
	// limit location
	if( agentX < 0.0f )
		agentX = 0.0f;
	else if( agentX > worldWidth )
		agentX = worldWidth;
	
	if( agentY < 0.0f )
		agentY = 0.0f;
	else if( agentY > worldHeight )
		agentY = worldHeight;
	
	//printf("fruit agent:  action %i - location %f %f - heading %f - velocity %f\n",
	//	   (int)action, agentX, agentY, agentDir, agentVel );
		   
	
	// check if agent has reached a goal
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t n=0; n < numFruit; n++ )
	{
		if( fruitObjects[n]->checkCollision(agentX, agentY, agentRad) )
		{
			if( reward != NULL )
				*reward = fruitObjects[n]->reward;
			
			Reset();
			return true;
		}
	}
	
	// check if the agent has exceeded the frame time limit
	const bool timeout = epFrameCount > epMaxFrames;
		
	if( reward != NULL )
		*reward = timeout ? -MAX_REWARD : 0.0f;
	
	epFrameCount++;
	
	if( timeout )
		Reset();
	
	return timeout;
}
	
		
// random_pos
void FruitEnv::randomize_pos( float* x, float* y )
{
	if( x != NULL )
		*x = randf(0.0f, worldWidth);
	
	if( y != NULL )
		*y = randf(0.0f, worldHeight);
}


// determine if a coordinate is inside (or on the border) of a circle
static inline bool check_inside( float x, float y, float cx, float cy, float radius )
{
	const float sx = x - cx;
	const float sy = y - cy;
	
	if( (sx * sx + sy * sy) <= (radius * radius) )
		return true;
	
	return false;
}


inline static void copy_color(float* src, float* dst)	{ dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3]; }


// Render
float* FruitEnv::Render()
{
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t y = 0; y < renderHeight; y++ )
	{
		for( uint32_t x = 0; x < renderWidth; x++ )
		{
			float* px = renderCPU + (y * renderWidth * 4 + x * 4);
	
			// check if this pixel is a fruit object
			bool is_fruit = false;
			
			for( uint32_t n = 0; n < numFruit; n++ )
			{
				if( check_inside(x, y, fruitObjects[n]->x, fruitObjects[n]->y, fruitObjects[n]->radius) )
				{
					copy_color(fruitObjects[n]->color, px);
					is_fruit = true;
				}
			}
			
			// check if this pixel is the agent, otherwise it's background
			if( !is_fruit )
			{
				if( check_inside(x, y, agentX, agentY, agentRad) )
					copy_color(agentColor, px);
				else
					copy_color(bgColor, px);
			}
		}
	}
	
	return renderCPU;
}
	

// Reset
void FruitEnv::Reset()
{
	// reset frame count
	epFrameCount = 0;
	
	// reset fruit positions
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t n = 0; n < numFruit; n++ )
		randomize_pos(&fruitObjects[n]->x, &fruitObjects[n]->y);
	
	// reset agent position
	randomize_pos(&agentX, &agentY);

	// check if there happen to be random overlap
	for( uint32_t n=0; n < numFruit; n++ )
		if( fruitObjects[n]->checkCollision(agentX, agentY, agentRad) )
			Reset();
}
