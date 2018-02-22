/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#ifndef __FRUIT_ENVIRONMENT_H__
#define __FRUIT_ENVIRONMENT_H__


#include <vector>
#include <stdint.h>


/**
 * Agent Actions
 */
enum AgentAction
{
	//ACTION_NONE = 0,
	ACTION_FORWARD = 0,
	ACTION_BACKWARD,
	ACTION_LEFT,
	ACTION_RIGHT,
	/* new actions */
	NUM_ACTIONS,
	ACTION_NONE
};


/**
 * Fruit Environment
 */
class FruitEnv
{
public:
	/**
	 * Create
	 */
	static FruitEnv* Create( uint32_t world_width, uint32_t world_height,
							 uint32_t max_episode_length=100 );
							 
	/**
	 * Create
	 */
	static FruitEnv* Create( uint32_t world_width, uint32_t world_height,
							 uint32_t render_width, uint32_t render_height,
							 uint32_t max_episode_length=100 );
	
	/**
	 * Destructor
	 */
	~FruitEnv();
	
	/**
	 * Perform the agent's next action
	 * @param[in] action the action selected for the agent to perform
	 * @param[out] reward the agent's reward as a result of the action
	 * @return true if EOE (End of Episode) has occurred
	 *         false if the action did not result in EOE
	 */
	bool Action( AgentAction action, float* reward );

	/**
	 * Convert an AgentAction enum to a string.
	 */
	static const char* ActionToStr( AgentAction action );
	
	/**
	 * Retrieve the scrolling width of the world, in pixels.
	 */
	inline uint32_t GetWorldWidth() const		{ return worldWidth; }
	
	/**
	 * Retrieve the scrolling height of the world, in pixels.
	 */
	inline uint32_t GetWorldHeight() const		{ return worldHeight; }
	
	/**
	 * Retrieve the width of the rendered image, in pixels.
	 */
	inline uint32_t GetRenderWidth() const		{ return renderWidth; }
	
	/**
	 * Retrieve the height of the rendered image, in pixels.
	 */
	inline uint32_t GetRenderHeight() const		{ return renderHeight; }

	/**
	 * Get the maximum reward value (signifies a win)
	 */
	inline float GetMaxReward() const			{ return MAX_REWARD; }
	
	/**
	 * Render the environment into an image.
	 * @returns pointer to float4 CUDA RGBA image of the rendered environment, 
	 *          with normalized pixel intensities 0.0f-1.0f
	 * @see GetRenderWidth for the width of the returned image pointer
	 * @see GetRenderHeight for the height of the returned image pointer
	 */
	float* Render();
	
	/**
	 * Reset environment
	 */
	void Reset();
	
private:
	FruitEnv();
	
	bool init( uint32_t world_width, uint32_t world_height,
			   uint32_t render_width, uint32_t render_height,
			   uint32_t episode_max_length );
			   
	static const int MAX_REWARD  = 1;	// max/min reward obtainable
	static const int MAX_OBJECTS = 1;	// max number of objects in world
	static const int MIX_OBJECTS = 50;  // mix of pos/neg objects (0-100%)
	static const int DEFAULT_RAD = 4;	// default radius of agent/fruit (in pixels)
	
	void randomize_pos( float* x, float* y );
	
	float agentX;		 // location of the agent (x-coordinate)
	float agentY;		 // location of the agent (y-coordinate)
	float agentDir;	 	 //  heading of the agent (0-359 degrees)
	float agentVelX;	 	 // velocity of the agent (-N to N)
	float agentVelY;
	float agentRad;		 //   radius of the agent (in pixels)
	float agentColor[4]; //    color of the agent (RGBA)
	float bgColor[4];	 // color of the background (RGBA)
	
	uint32_t epMaxFrames;	// maximum number of frames per episode
	uint32_t epFrameCount;	// frame counter for current episode
	uint32_t worldWidth; 	// width of the environment (in pixels)
	uint32_t worldHeight; 	// height of the environment (in pixels)
	uint32_t renderWidth;	// width of the output image (in pixels)
	uint32_t renderHeight;	// height of the output image (in pixels)
	
	float* renderCPU;
	float* renderGPU;
	
	// fruit objects
	struct fruitObject
	{
		float x;
		float y;
		float reward;
		float radius;
		float color[4];
		
		inline fruitObject()
		{
			x = 0.0f;
			y = 0.0f;
			
			reward = MAX_REWARD;
			radius = DEFAULT_RAD;
			
			color[0] = 1.0f;
			color[1] = 1.0f;
			color[2] = 0.0f;
			color[3] = 1.0f;
		}
		
		inline bool checkCollision( float obj_x, float obj_y, float obj_radius )	
		{ 
			const float sx = x - obj_x;
			const float sy = y - obj_y;
			const float s2 = sx * sx + sy * sy;
			
			const float r0 = radius - obj_radius;
			const float r1 = radius + obj_radius;
			
			// (R0-R1)^2 <= (x0-x1)^2+(y0-y1)^2 <= (R0+R1)^2
			return ((r0 * r0) <= s2) && (s2 <= (r1 * r1)); 
		}

		inline float distanceSq( float obj_x, float obj_y )
		{
			const float sx = x - obj_x;
			const float sy = y - obj_y;

			return sx * sx + sy * sy;
		}
	};
	
	std::vector<fruitObject*> fruitObjects;	   // list of objects in the environment

	// return the closest fruit to the agent, return the object and distance squared
	fruitObject* findClosest( float* distanceSq ) const;	

	float lastDistanceSq;
	float spawnDistanceSq;
};

#endif
