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
class fruitEnv
{
public:
	/**
	 * Create
	 */
	static fruitEnv* Create( int env_width=512, int env_height=512 );
	
	/**
	 * Destructor
	 */
	~fruitEnv();
	
	/**
	 * Perform the agent's next action
	 * @param[in] action the action selected for the agent to perform
	 * @param[out] reward the agent's reward as a result of the action
	 * @return true if EOE (End of Episode) has occurred
	 *         false if the action did not result in EOE
	 */
	bool Action( AgentAction action, float* reward );
	
	/**
	 * Render
	 */
	bool Render( uint32_t width, uint32_t height );
	
	/**
	 * Reset environment
	 */
	void Reset();
	
private:
	fruitEnv();
	bool init( int env_width, int env_height );
	
	int agentX;		// location of the agent (x-coordinate)
	int agentY;		// location of the agent (y-coordinate)
	int agentDir;	//  heading of the agent (0-359 degrees)
	int agentVel;	// velocity of the agent (-N to N)
	
	int envWidth;	// width of the environment (in pixels)
	int envHeight;	// height of the environment (in pixels)
	
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
