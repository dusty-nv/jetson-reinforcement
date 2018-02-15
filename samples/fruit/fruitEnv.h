/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */
 
#ifndef __FRUIT_ENVIRONMENT_H__
#define __FRUIT_ENVIRONMENT_H__


#include <vector>


/**
 * Agent Actions
 */
enum AgentAction
{
	ACTION_NONE = 0,
	ACTION_FORWARD,
	ACTION_BACKWARD,
	ACTION_LEFT,
	ACTION_RIGHT,
	/* new actions */
	NUM_ACTIONS
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
	static FruitEnv* Create( uint32_t world_width=512, uint32_t world_height=512,
							 uint32_t render_width=512, uint32_t render_height=512 );
	
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
	 * Retrieve the scrolling width of the world, in pixels.
	 */
	inline uint32_t GetWorldWidth() const		{ return worldWidth; }
	
	/**
	 * Retrieve the scrolling height of the world, in pixels.
	 */
	inline uint32_t GetRenderHeight() const		{ return worldHeight; }
	
	/**
	 * Retrieve the width of the rendered image, in pixels.
	 */
	inline uint32_t GetRenderWidth() const		{ return renderWidth; }
	
	/**
	 * Retrieve the height of the rendered image, in pixels.
	 */
	inline uint32_t GetRenderHeight() const		{ return renderHeight; }
	
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
	bool init( int env_width, int env_height );
	
	int agentX;		 // location of the agent (x-coordinate)
	int agentY;		 // location of the agent (y-coordinate)
	int agentDir;	 //  heading of the agent (0-359 degrees)
	int agentVel;	 // velocity of the agent (-N to N)
	
	int worldWidth;	 // width of the environment (in pixels)
	int worldHeight; // height of the environment (in pixels)
	int renderWidth;
	int renderHeight;
	
	float* imageCPU;
	float* imageGPU;
	
	// fruit objects
	struct envObject
	{
		int x;
		int y;
		int reward;
	};
	
	std::vector<envObject*> envObjects;	// list of objects in the environment
	
	static const int MAX_REWARD  = 100;	// max/min reward obtainable
	static const int MAX_OBJECTS = 10;	// max number of objects in world
	static const int MIX_OBJECTS = 50;  // mix of pos/neg objects (0-100%)
};

#endif
