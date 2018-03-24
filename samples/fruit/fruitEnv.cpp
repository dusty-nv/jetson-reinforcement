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
	if( action == ACTION_FORWARD )		return "DOWN ";	// up/down are reversed visually
	else if( action == ACTION_BACKWARD )	return "UP   ";	// in the GUI due to y-coordinate
	else if( action == ACTION_LEFT )		return "LEFT ";
	else if( action == ACTION_RIGHT )		return "RIGHT";
	else if( action == ACTION_NONE )		return "NONE ";

	return "NULL ";
}


// Constructor
FruitEnv::FruitEnv()
{
	agentX   = 0;
	agentY   = 0;
	agentDir = 0;
	//agentVel = 0;
	agentRad = DEFAULT_RAD;
	agentVelX = 0;
	agentVelY = 0;

	agentColor[0] = 1.0f; 
	agentColor[1] = 0.0f; 
	agentColor[2] = 1.0f; 
	agentColor[3] = 1.0f;
	
	bgColor[0] = 0.0f; 
	bgColor[1] = 0.0f; 
	bgColor[2] = 0.0f; 
	bgColor[3] = 0.0f;
	
	epMaxFrames  = 0;
	epFrameCount = 0;
	worldWidth   = 0;
	worldHeight  = 0;
	renderWidth  = 0;
	renderHeight = 0;

	lastDistanceSq  = 0;
	spawnDistanceSq = 0;
	
	renderCPU = NULL;
	renderGPU = NULL;
	
	srand_time();	// seed rng
}


// Destructor
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


// Initialize enviroment
bool FruitEnv::init( uint32_t world_width, uint32_t world_height, uint32_t render_width, uint32_t render_height, uint32_t episode_max_length )
{
	worldWidth   = world_width;
	worldHeight  = world_height;
	renderWidth  = render_width;
	renderHeight = render_height;
	epMaxFrames  = episode_max_length;
	
	// Allocate render image memory
	if( !cudaAllocMapped((void**)&renderCPU, (void**)&renderGPU, renderWidth * renderHeight * sizeof(float) * 4) )
		return false;
	
	// Allocate fruit objects
	for( uint32_t n=0; n < MAX_OBJECTS; n++ )
	{
		fruitObject* obj = new fruitObject();
		fruitObjects.push_back(obj);
	}
	
	// Reset the first time
	Reset();

	// Print some settings
	printf("[deepRL]  ep_max_frames:  %u\n", epMaxFrames);
	return true;
}


// Action
bool FruitEnv::Action( AgentAction action, float* reward )
{	
	// first, make sure the action is valid
	/*if( action >= NUM_ACTIONS )
	{
		printf("FruitEnv::Action() -- invalid action selected (%i)\n", (int)action);
		return false;
	}*/
	
//#define FIRST_ORDER
#ifdef FIRST_ORDER
	// Apply action
	const float delta = 1.0f;

	if( action == ACTION_FORWARD )
		agentY -= delta;
	else if( action == ACTION_BACKWARD )
		agentY += delta;
	else if( action == ACTION_RIGHT )
		agentX += delta;
	else if( action == ACTION_LEFT )
		agentX -= delta;
#else
	const float vel_delta = 0.5f;
	const float dir_delta = 2.5f;

	if( action == ACTION_FORWARD )
		agentVelY += vel_delta;
	else if( action == ACTION_BACKWARD )
		agentVelY -= vel_delta;
	else if( action == ACTION_RIGHT )
		agentVelX += vel_delta;
	else if( action == ACTION_LEFT )
		agentVelX -= vel_delta;
	
	// Limit velocity
	const float maxVelocity = 0.5f;	
	
	if( agentVelX < -maxVelocity )
		agentVelX = -maxVelocity;
	else if( agentVelX > maxVelocity )
		agentVelX = maxVelocity;

	if( agentVelY < -maxVelocity )
		agentVelY = -maxVelocity;
	else if( agentVelY > maxVelocity )
		agentVelY = maxVelocity;
	
	agentX += agentVelX;
	agentY += agentVelY;
#endif

	
	// Limit location
	bool outOfBounds = false;

	if( agentX < 0.0f )
	{
		agentX = 0.0f;
		outOfBounds = true;
	}
	else if( agentX > worldWidth )
	{
		agentX = worldWidth;
		outOfBounds = true;
	}

	if( agentY < 0.0f )
	{
		agentY = 0.0f;
		outOfBounds = true;
	}
	else if( agentY > worldHeight )
	{
		agentY = worldHeight;
		outOfBounds = true;
	}


	// If the agent went out of bounds, it counts as a loss
	if( outOfBounds )
	{
		Reset();	// end of episode, reset

		if( reward != NULL )
			*reward = -MAX_REWARD;

		return true;
	}
   
	//printf("fruit agent:  action %i - location %f %f - heading %f - velocity %f\n",
	//	   (int)action, agentX, agentY, agentDir, agentVel );
		

	// Check if agent has reached a goal
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t n=0; n < numFruit; n++ )
	{
		if( fruitObjects[n]->checkCollision(agentX, agentY, agentRad) )
		{
			if( reward != NULL )
				*reward = fruitObjects[n]->reward;
			
			Reset();	// end of episode, reset
			return true;
		}
	}
	
	// Check if the agent has exceeded the frame time limit
	const bool timeout = epFrameCount > epMaxFrames;
	
	if( timeout )
	{
		Reset();	// end of episode, reset

		if( reward != NULL )
			*reward = -MAX_REWARD;
	}
	else
	{
		epFrameCount++;	// increment frame count for next time

		// Compute the reward based on how close it is to the closest fruit
		float         fruitDistSq = 0.0f;
		fruitObject* closestFruit = findClosest(&fruitDistSq);

		if( reward != NULL )
			//*reward = (lastDistanceSq > fruitDistSq) ? 1.0f : 0.0f;
			//*reward = (sqrtf(lastDistanceSq) - sqrtf(fruitDistSq)) * 0.5f;
			//*reward = (1.0f - (fruitDistSq / spawnDistanceSq)) * 0.1f;
			*reward = (sqrtf(lastDistanceSq) - sqrtf(fruitDistSq)) * 0.35f;
			//*reward = 1.0f - (1.0f/(1.0f + (fruitDistSq / float(worldWidth*worldWidth))));
			//*reward = exp(-(fruitDistSq/worldWidth/1.5f));
			//*reward = (exp(fruitDistSq)-1.0f)/sqrtf(pow(worldWidth,2));

		lastDistanceSq = fruitDistSq;
	}

	return timeout;
}
	

// Find closest object to agent
FruitEnv::fruitObject* FruitEnv::findClosest( float* distanceOut ) const
{
	fruitObject* min_obj = NULL;
	float       min_dist = 0.0f;
	const size_t num_obj = fruitObjects.size();

	for( size_t n=0; n < num_obj; n++ )
	{
		const float new_dist = fruitObjects[n]->distanceSq(agentX, agentY);

		if( !min_obj || new_dist < min_dist )
		{
			min_obj  = fruitObjects[n];
			min_dist = new_dist;
		}
	}

	if( distanceOut != NULL )
		*distanceOut = min_dist;

	return min_obj;
}

	
// Random_pos
void FruitEnv::randomize_pos( float* x, float* y )
{
	if( x != NULL )
		*x = randf(0.0f, worldWidth);
	
	if( y != NULL )
		*y = randf(0.0f, worldHeight);
}


// Determine if a coordinate is inside (or on the border) of a circle
static inline bool check_inside( float x, float y, float cx, float cy, float radius )
{
	const float sx = x - cx;
	const float sy = y - cy;
	
	if( (sx * sx + sy * sy) <= (radius * radius) )
		return true;
	
	return false;
}


// Copy RGBA colors
inline static void copy_color( float* src, float* dst )	
{ 
	dst[0] = src[0]; 
	dst[1] = src[1]; 
	dst[2] = src[2]; 
	dst[3] = src[3]; 
}


// Render
float* FruitEnv::Render()
{
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t y = 0; y < renderHeight; y++ )
	{
		for( uint32_t x = 0; x < renderWidth; x++ )
		{
			float* px = renderCPU + (y * renderWidth * 4 + x * 4);
	
			// Check if this pixel is a fruit object
			bool is_fruit = false;
			
			for( uint32_t n = 0; n < numFruit; n++ )
			{
				if( check_inside(x, y, fruitObjects[n]->x, fruitObjects[n]->y, fruitObjects[n]->radius) )
				{
					copy_color(fruitObjects[n]->color, px);
					is_fruit = true;
				}
			}
			
			// Check if this pixel is the agent, otherwise it's background
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
	// Reset frame count
	epFrameCount = 0;
	
	// Reset fruit positions
	const uint32_t numFruit = fruitObjects.size();
	
	for( uint32_t n = 0; n < numFruit; n++ )
	{
		//randomize_pos(&fruitObjects[n]->x, &fruitObjects[n]->y);
		fruitObjects[n]->x = float(worldWidth) * 0.5f;
		fruitObjects[n]->y = float(worldHeight) * 0.5f;
	}
	
	// Reset agent dynamics
	agentDir = 0;
	// agentVel = 0;
	agentVelX = 0;
	agentVelY = 0;

	// Reset agent position
	randomize_pos(&agentX, &agentY);
	//agentX = float(worldWidth) * 0.5f;
	//agentY = float(worldHeight) * 0.5f;

	// Check if there happen to be random overlap
	for( uint32_t n=0; n < numFruit; n++ )
		if( fruitObjects[n]->checkCollision(agentX, agentY, agentRad) )
			Reset();

	findClosest(&lastDistanceSq);
	spawnDistanceSq = lastDistanceSq;
	//printf("RESET -- lastDistanceSq = %f\n", lastDistanceSq);
}
