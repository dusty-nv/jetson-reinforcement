/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "RoverPlugin.h"
#include "PropPlugin.h"
#include "GazeboUtils.h"

#include "cudaMappedMemory.h"
#include "cudaPlanar.h"


#define PI 3.141592653589793238462643383279502884197169f

#define LF_HINGE "rover::front_left_wheel_hinge"
#define LB_HINGE "rover::back_left_wheel_hinge"
#define RF_HINGE "rover::front_right_wheel_hinge"
#define RB_HINGE "rover::back_right_wheel_hinge"

#define JOINT_MIN	-0.75f
#define JOINT_MAX	 2.0f

#define VELOCITY_MIN -1.0f
#define VELOCITY_MAX  1.0f

// Define DQN API Settings
#define INPUT_WIDTH   64
#define INPUT_HEIGHT  64
#define INPUT_CHANNELS 3
#define OPTIMIZER "RMSprop"
#define LEARNING_RATE 0.1f
#define REPLAY_MEMORY 10000
#define BATCH_SIZE 32
#define GAMMA 0.9f
#define EPS_START 0.9f
#define EPS_END 0.05f
#define EPS_DECAY 200
#define USE_LSTM true
#define LSTM_SIZE 256
#define ALLOW_RANDOM true
#define DEBUG_DQN false

#define NET_OUTPUTS 4

#define WORLD_NAME "rover_world"
#define ROVER_NAME "rover"
#define GOAL_NAME  "goal"

#define REWARD_WIN   500.0f
#define REWARD_LOSS -500.0f
#define REWARD_MULTIPLIER 100.0f

#define COLLISION_FILTER "ground_plane::link::collision"

#define ANIMATION_STEPS 2000


namespace gazebo
{
 
// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(RoverPlugin)


// constructor
RoverPlugin::RoverPlugin() : ModelPlugin(), cameraNode(new gazebo::transport::Node()), collisionNode(new gazebo::transport::Node())
{
	printf("RoverPlugin::RoverPlugin()\n");

	agent 	        = NULL;
	opMode		   = AGENT_LEARN;
	inputState        = NULL;
	inputBuffer[0]    = NULL;
	inputBuffer[1]    = NULL;
	inputBufferSize   = 0;
	inputRawWidth     = 0;
	inputRawHeight    = 0;
	actionVelDelta    = 0.1f;
	maxEpisodeLength  = 200;	// set to # frames to limit ep length
	episodeFrames     = 0;
	episodesCompleted = 0;
	episodesWon	   = 0;

	newState          = false;
	newReward         = false;
	endEpisode        = false;
	rewardHistory     = 0.0f;
	lastGoalDistance  = 0.0f;
	avgGoalDelta	   = 0.0f;
	lastAction        = -1;
	runHistoryIdx     = 0;
	runHistoryMax     = 0;

	memset(runHistory, 0, sizeof(runHistory));

	for( uint32_t n=0; n < DOF; n++ )
		vel[n] = 0.0f;

	HID = NULL;
}


// configJoint 
bool RoverPlugin::configJoint( const char* name )
{
	std::vector<physics::JointPtr> jnt = model->GetJoints();
	const size_t numJoints = jnt.size();

	// find the joint with the specified name
	for( uint32_t n=0; n < numJoints; n++ )
	{
		if( strcmp(name, jnt[n]->GetScopedName().c_str()) == 0 )
		{
			jnt[n]->SetVelocity(0, 0.0);
			joints.push_back(jnt[n]);
			return true;
		}
	}

	printf("RoverPlugin -- failed to find joint '%s'\n", name);
	return false;
}


// Load
void RoverPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) 
{
	printf("RoverPlugin::Load('%s')\n", _parent->GetName().c_str());

	// Store the pointer to the model
	this->model = _parent;
	
	// Configure the drive joints
	configJoint(LF_HINGE);
	configJoint(LB_HINGE);
	configJoint(RF_HINGE);
	configJoint(RB_HINGE);

	// Store the original pose of the model
	this->originalPose = model->GetWorldPose();

	// create DQN agent
	if( !createAgent() )
		printf("RoverPlugin -- failed to create DQN agent\n");

	// Create our node for camera communication
	cameraNode->Init();
	cameraSub = cameraNode->Subscribe("/gazebo/" WORLD_NAME "/" ROVER_NAME "/camera_link/camera/image", &RoverPlugin::onCameraMsg, this);

	// Create our node for collision detection
	collisionNode->Init();
	collisionSub = collisionNode->Subscribe("/gazebo/" WORLD_NAME "/" ROVER_NAME "/chassis/my_contact", &RoverPlugin::onCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&RoverPlugin::OnUpdate, this, _1));
}


// onCameraMsg
void RoverPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
	static int iter = 0;
	iter++;
	
	if( iter < 10 )
		return;


	// check the validity of the message contents
	if( !_msg )
	{
		printf("RoverPlugin - recieved NULL message\n");
		return;
	}

	// retrieve image dimensions
	const int width  = _msg->image().width();
	const int height = _msg->image().height();
	const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;	// bits per pixel
	const int size   = _msg->image().data().size();

	if( bpp != 24 )
	{
		printf("RoverPlugin - expected 24BPP uchar3 image from camera, got %i\n", bpp);
		return;
	}

	// allocate temp image if necessary
	if( !inputBuffer[0] || size != inputBufferSize )
	{
		if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
		{
			printf("RoverPlugin - cudaAllocMapped() failed to allocate %i bytes\n", size);
			return;
		}

		printf("RoverPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n", width, height, bpp, size);
		
		inputBufferSize = size;
		inputRawWidth   = width;
		inputRawHeight  = height;
	}

	memcpy(inputBuffer[0], _msg->image().data().c_str(), inputBufferSize);
	newState = true;
	//printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);
}


// onCollisionMsg
void RoverPlugin::onCollisionMsg(ConstContactsPtr &contacts)
{
	//printf("collision callback (%u contacts)\n", contacts->contact_size());

	for (unsigned int i = 0; i < contacts->contact_size(); ++i)
	{
		if( strcmp(contacts->contact(i).collision2().c_str(), COLLISION_FILTER) == 0 )
			continue;

		std::cout << "Collision between[" << contacts->contact(i).collision1()
			     << "] and [" << contacts->contact(i).collision2() << "]\n";


		for (unsigned int j = 0; j < contacts->contact(i).position_size(); ++j)
		{
			 std::cout << j << "  Position:"
					 << contacts->contact(i).position(j).x() << " "
					 << contacts->contact(i).position(j).y() << " "
					 << contacts->contact(i).position(j).z() << "\n";
			 std::cout << "   Normal:"
					 << contacts->contact(i).normal(j).x() << " "
					 << contacts->contact(i).normal(j).y() << " "
					 << contacts->contact(i).normal(j).z() << "\n";
			 std::cout << "   Depth:" << contacts->contact(i).depth(j) << "\n";
		}

		// issue learning reward
		if( opMode == AGENT_LEARN )
		{
			#define GOAL_COLLISION "goal::link::box_collision"

			bool hitTarget = (strcmp(contacts->contact(i).collision2().c_str(), GOAL_COLLISION) == 0) ||
						  (strcmp(contacts->contact(i).collision1().c_str(), GOAL_COLLISION) == 0);

			rewardHistory = hitTarget ? REWARD_WIN : REWARD_LOSS;

			newReward  = true;
			endEpisode = true;
		}
	}
}


// createAgent
bool RoverPlugin::createAgent()
{
	if( agent != NULL )
		return true;

	// Create AI agent
	agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, NET_OUTPUTS, 
						OPTIMIZER, LEARNING_RATE, REPLAY_MEMORY, BATCH_SIZE,
						GAMMA, EPS_START, EPS_END, EPS_DECAY, 
						USE_LSTM, LSTM_SIZE, ALLOW_RANDOM, DEBUG_DQN);
	if( !agent )
	{
		printf("RoverPlugin - failed to create AI agent\n");
		return false;
	}

	inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

	if( !inputState )
	{
		printf("RoverPlugin - failed to allocate %ux%ux%u Tensor\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);
		return false;
	}

	return true;
}


// upon recieving a new frame, update the AI agent
bool RoverPlugin::updateAgent()
{
	if( !agent )
		return false;

	// convert uchar3 input from camera to planar BGR
	if( CUDA_FAILED(cudaPackedToPlanarBGR((uchar3*)inputBuffer[1], inputRawWidth, inputRawHeight,
							         inputState->gpuPtr, INPUT_WIDTH, INPUT_HEIGHT)) )
	{
		printf("RoverPlugin - failed to convert %zux%zu image to %ux%u planar BGR image\n",
			   inputRawWidth, inputRawHeight, INPUT_WIDTH, INPUT_HEIGHT);

		return false;
	}

	// select the next action
	int action = 0;

	if( !agent->NextAction(inputState, &action) )
	{
		printf("RoverPlugin - failed to generate agent's next action\n");
		return false;
	}

	// make sure the selected action is in-bounds
	if( action < 0 || action >= /*DOF * 2 + 1*/ NET_OUTPUTS )
	{
		printf("RoverPlugin - agent selected invalid action, %i\n", action);
		return false;
	}

	//printf("RoverPlugin - agent selected action %i\n", action);

#if 0
	// action 0 does nothing, the others index a joint
	if( action == 0 )
		return false;	// not an error, but didn't cause an update
	
	action--;	// with action 0 = no-op, index 1 should map to joint 0

	// if the action is even, increase the joint position by the delta parameter
	// if the action is odd,  decrease the joint position by the delta parameter
	vel[action/2] = vel[action/2] + actionVelDelta * ((action % 2 == 0) ? 1.0f : -1.0f);
#endif

	if( action == 0 )
	{
		for( uint32_t n=0; n < DOF; n++ )
			vel[n] = VELOCITY_MIN;
	}
	else if( action == 1 )
	{
		for( uint32_t n=0; n < DOF; n++ )
			vel[n] = VELOCITY_MAX;
	}
	else if( action == 2 )
	{
		vel[0] = VELOCITY_MIN;
		vel[1] = VELOCITY_MAX;
	}
	else if( action == 3 )
	{
		vel[0] = VELOCITY_MAX;
		vel[1] = VELOCITY_MIN;
	}

	lastAction = action;
	return true;
}


// return the tetxual string corresponding to an action
static const char* actionStr( int action )
{
	if( action == 0 )		return "REV  ";
	else if( action == 1 )	return "FWD  ";
	else if( action == 2 )	return "LEFT ";
	else if( action == 3 )	return "RIGHT";
	else					return "NULL ";
}


// update joint reference positions, returns true if positions have been modified
bool RoverPlugin::updateJoints()
{
	if( opMode == USER_MANUAL )	
	{
		// make sure the HID interface is open
		if( !HID )
		{
			static int count = 0;	// BUG:  gazebo glitches when dev opened early
	
			if( count > 1000 )
				HID = InputDevices::Create();

			count++;

			if( !HID )
				return false;	// TODO: print Try running sudo?
		}
		
		// poll for input events
		HID->Poll();

		// retrieve keyboard device
		KeyboardDevice* keyboard = HID->GetKeyboard();

		if( !keyboard )
			return false;

		if( keyboard->KeyDown(KEY_W) )
		{
			vel[0] += actionVelDelta;
			printf("KEY_W\n");
		}
		if( keyboard->KeyDown(KEY_S) )
		{
			vel[0] -= actionVelDelta;
			printf("KEY_S\n");
		}
		if( keyboard->KeyDown(KEY_I) )
		{
			vel[1] += actionVelDelta;
			printf("KEY_I\n");
		}
		if( keyboard->KeyDown(KEY_K) )
		{
			vel[1] -= actionVelDelta;
			printf("KEY_K\n");
		}

		return true;
	}
	else if( newState )
	{
		// update the AI agent when new camera frame is ready
		episodeFrames++;
		//printf("episode %i frame = %i\n", episodesCompleted, episodeFrames);

		// reset camera ready flag
		newState = false;

		if( updateAgent() )
			return true;
	}

	return false;
}



// called by the world update start event
void RoverPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
{
	static int iter = 0;
	iter++;
	
	if( iter < 1000 )
		return;

	const bool hadNewState = newState && (opMode == AGENT_LEARN);

	// update the robot positions with vision/DQN
	if( updateJoints() )
	{
		for( int i=0; i < DOF; i++ )
		{
			if( vel[i] < VELOCITY_MIN )
				vel[i] = VELOCITY_MIN;

			if( vel[i] > VELOCITY_MAX )
				vel[i] = VELOCITY_MAX;
		}

		if( joints.size() != 4 )
		{
			printf("RoverPlugin -- could only find %zu of 4 drive joints\n", joints.size());
			return;
		}

		joints[0]->SetVelocity(0, vel[0]);
		joints[1]->SetVelocity(0, vel[0]);
		joints[2]->SetVelocity(0, vel[1]);
		joints[3]->SetVelocity(0, vel[1]);
	}

	// episode timeout
	if( maxEpisodeLength > 0 && episodeFrames > maxEpisodeLength )
	{
		printf("RoverPlugin - triggering EOE, episode has exceeded %i frames\n", maxEpisodeLength);

		rewardHistory = REWARD_LOSS;
		newReward     = true;
		endEpisode    = true;
	}

	// if an EOE reward hasn't already been issued, compute one
	if( hadNewState && !newReward )
	{
		PropPlugin* goal = GetPropByName(GOAL_NAME);

		if( !goal )
		{
			printf("RoverPlugin - failed to find Prop '%s'\n", GOAL_NAME);
			return;
		}

		// get the bounding box for the prop object
		const math::Box& goalBBox = goal->model->GetBoundingBox();
		physics::LinkPtr chassis  = model->GetLink("chassis");

		if( !chassis )
		{
			printf("RoverPlugin - failed to find chassis link\n");
			return;
		}

		const math::Box& chassisBBox = chassis->GetBoundingBox();
		const float distGoal  = BoxDistance(goalBBox, chassisBBox); 
		const float distDelta = lastGoalDistance - distGoal;
		const float movingAvg = 0.0f;	// rolling average to smooth out the intermediary reward and prevent "hovering"

		avgGoalDelta  = (avgGoalDelta * movingAvg) + (distDelta * (1.0f - movingAvg));
		rewardHistory = (episodeFrames > 1) ? avgGoalDelta * REWARD_MULTIPLIER : 0.0f;

		if( distGoal < 0.01 )
			rewardHistory = REWARD_WIN;	// TODO investigate if necessary

		lastGoalDistance = distGoal;

		/*bool isMoving = false;

		for( uint32_t n=0; n < DOF; n++ )
			if( vel[n] != 0.0f )	// TODO epsilon of actionVelDelta?
				isMoving = true;

		if( isMoving && episodeFrames % 10 == 0 )
			rewardHistory = 1.0f;	// the rover hasn't run into anything this frame (pos reward)
		else
			rewardHistory = 0.0f;*/

		newReward = true;	
	}

	// issue rewards and train DQN
	if( newReward && agent != NULL )
	{
		//printf("RoverPlugin - issuing reward %f, EOE=%s  %s\n", rewardHistory, endEpisode ? "true" : "false", (rewardHistory > 0.1f) ? "POS+" : (rewardHistory > 0.0f) ? "POS" : (rewardHistory < 0.0f) ? "    NEG" : "       ZERO");
		agent->NextReward(rewardHistory, endEpisode);

		// reset reward indicator
		newReward = false;

		// reset for next episode
		if( endEpisode )
		{
			endEpisode       = false;
			episodeFrames    = 0;
			lastGoalDistance = 0.0f;
			avgGoalDelta     = 0.0f;

			// reset simulation state
			// ResetPropDynamics();  // DISABLED:  turns out the rover doesn't move the box anyways (now the user can move the box themselves)

			for( uint32_t n=0; n < DOF; n++ )
				vel[n] = 0.0f;

			model->SetAngularAccel(math::Vector3(0.0, 0.0, 0.0));
			model->SetAngularVel(math::Vector3(0.0, 0.0, 0.0));
			model->SetLinearAccel(math::Vector3(0.0, 0.0, 0.0));
			model->SetLinearVel(math::Vector3(0.0, 0.0, 0.0));

			model->SetWorldPose(originalPose);

			// tally episode stats
			if( rewardHistory >= REWARD_WIN )
			{
				runHistory[runHistoryIdx] = true;
				episodesWon++;
			}
			else
				runHistory[runHistoryIdx] = false;

			runHistoryIdx = (runHistoryIdx + 1) % sizeof(runHistory);
			episodesCompleted++;
		}

		// print agent stats
		printf("action = %s reward = %+0.4f  wins = %03u of %03u (%0.2f)   ", 
			  actionStr(lastAction), rewardHistory,  
			  episodesWon, episodesCompleted, float(episodesWon)/float(episodesCompleted));

		const uint32_t RUN_HISTORY = sizeof(runHistory);

		if( episodesCompleted >= RUN_HISTORY )
		{
			uint32_t historyWins = 0;

			for( uint32_t n=0; n < RUN_HISTORY; n++ )
			{
				if( runHistory[n] )
					historyWins++;
			}

			if( historyWins > runHistoryMax )
				runHistoryMax = historyWins;

			printf("%02u of last %u  (%0.2f)  (max=%0.2f)", historyWins, RUN_HISTORY, float(historyWins)/float(RUN_HISTORY), float(runHistoryMax)/float(RUN_HISTORY));
		}

		printf("\n");
	}
}

}

