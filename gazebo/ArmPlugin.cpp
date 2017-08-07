/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "ArmPlugin.h"

#include "cudaMappedMemory.h"
#include "cudaPlanar.h"


#define PI 3.141592653589793238462643383279502884197169f

#define ANIMATION_STEPS 2000
#define JOINT_MIN	-0.75f
#define JOINT_MAX	 0.75f

#define INPUT_WIDTH   40
#define INPUT_HEIGHT  80
#define INPUT_CHANNELS 3

#define COLLISION_FILTER "ground_plane::link::collision"


namespace gazebo
{
 
// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(ArmPlugin)


// constructor
ArmPlugin::ArmPlugin() : ModelPlugin(), cameraNode(new gazebo::transport::Node()), collisionNode(new gazebo::transport::Node())
{
	printf("ArmPlugin::ArmPlugin()\n");

	for( uint32_t n=0; n < DOF; n++ )
		ref[n] = JOINT_MIN;

	agent 	       = NULL;
	inputState       = NULL;
	inputBuffer[0]   = NULL;
	inputBuffer[1]   = NULL;
	inputBufferSize  = 0;
	inputRawWidth    = 0;
	inputRawHeight   = 0;
	actionJointDelta = 0.1f;

	newState         = false;
	testAnimation    = false;
	animationStep    = 0;
}


// Load
void ArmPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) 
{
	printf("ArmPlugin::Load()\n");

	// Create AI agent
	agent = dqnAgent::Create(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS, DOF*2+1);

	if( !agent )
		printf("ArmPlugin - failed to create AI agent\n");

	inputState = Tensor::Alloc(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

	if( !inputState )
		printf("ArmPlugin - failed to allocate %ux%ux%u Tensor\n", INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS);

	// Store the pointer to the model
	this->model = _parent;
	this->j2_controller = new physics::JointController(model);

	// Create our node for camera communication
	cameraNode->Init();
	cameraSub = cameraNode->Subscribe("/gazebo/default/camera/link/camera/image", &ArmPlugin::onCameraMsg, this);

	// Create our node for collision detection
	collisionNode->Init();
	collisionSub = collisionNode->Subscribe("/gazebo/default/box/link/my_contact", &ArmPlugin::onCollisionMsg, this);

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&ArmPlugin::OnUpdate, this, _1));
}


// onCameraMsg
void ArmPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
	// check the validity of the message contents
	printf("camera callback\n");

	if( !_msg )
	{
		printf("ArmPlugin - recieved NULL message\n");
		return;
	}

	// retrieve image dimensions
	const int width  = _msg->image().width();
	const int height = _msg->image().height();
	const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;	// bits per pixel
	const int size   = _msg->image().data().size();

	if( bpp != 24 )
	{
		printf("ArmPlugin - expected 24BPP uchar3 image from camera, got %i\n", bpp);
		return;
	}

	// allocate temp image if necessary
	if( !inputBuffer[0] || size != inputBufferSize )
	{
		if( !cudaAllocMapped(&inputBuffer[0], &inputBuffer[1], size) )
		{
			printf("ArmPlugin - cudaAllocMapped() failed to allocate %i bytes\n", size);
			return;
		}

		printf("ArmPlugin - allocated camera img buffer %ix%i  %i bpp  %i bytes\n", width, height, bpp, size);
		
		inputBufferSize = size;
		inputRawWidth   = width;
		inputRawHeight  = height;
	}

	memcpy(inputBuffer[0], _msg->image().data().c_str(), inputBufferSize);
	newState = true;

	/* unsigned int oldCount = this->data.image_count;
	this->data.image_count = _msg->image().data().size();

	if (oldCount != this->data.image_count)
	{
		delete this->data.image;
		this->data.image = new uint8_t[this->data.image_count];
	}

	// Set the image pixels
	memcpy(this->data.image, _msg->image().data().c_str(),_msg->image().data().size());

	size = sizeof(this->data) - sizeof(this->data.image) +
	_msg->image().data().size(); */

	printf("camera %i x %i  %i bpp  %i bytes\n", width, height, bpp, size);
	//std::cout << _msg->DebugString();
}


// onCollisionMsg
void ArmPlugin::onCollisionMsg(ConstContactsPtr &contacts)
{
	//printf("collision callback\n");

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
	}
}


// upon recieving a new frame, update the AI agent
bool ArmPlugin::updateAgent()
{
	// convert uchar3 input from camera to planar BGR
	if( CUDA_FAILED(cudaPackedToPlanarBGR((uchar3*)inputBuffer[1], inputRawWidth, inputRawHeight,
							         inputState->gpuPtr, INPUT_WIDTH, INPUT_HEIGHT)) )
	{
		printf("ArmPlugin - failed to convert %zux%zu image to %ux%u planar BGR image\n",
			   inputRawWidth, inputRawHeight, INPUT_WIDTH, INPUT_HEIGHT);

		return false;
	}

	// select the next action
	int action = 0;

	if( !agent->NextAction(inputState, &action) )
	{
		printf("ArmPlugin - failed to generate agent's next action\n");
		return false;
	}

	// make sure the selected action is in-bounds
	if( action < 0 || action >= DOF * 2 + 1 )
	{
		printf("ArmPlugin - agent selected invalid action, %i\n", action);
		return false;
	}

	printf("ArmPlugin - agent selected action %i\n", action);

	// action 0 does nothing, the others index a joint
	if( action == 0 )
		return false;	// not an error, but didn't cause an update
	
	action--;	// with action 0 = no-op, index 1 should map to joint 0

	// if the action is even, increase the joint position by the delta parameter
	// if the action is odd,  decrease the joint position by the delta parameter
	float joint = ref[action/2] + actionJointDelta * ((action % 2 == 0) ? 1.0f : -1.0f);

	// limit the joint to the specified range
	if( joint < JOINT_MIN )
		joint = JOINT_MIN;
	
	if( joint > JOINT_MAX )
		joint = JOINT_MAX;

	ref[action/2] = joint;
	return true;
}


// update joint reference positions, returns true if positions have been modified
bool ArmPlugin::updateJoints()
{
	// update the AI agent if a new camera frame is ready
	if( newState && agent != NULL )
	{
		newState = false;

		if( updateAgent() )
			return true;
	}
	else if( testAnimation )	// test sequence
	{
		const float step = (JOINT_MAX - JOINT_MIN) * (float(1.0f) / float(ANIMATION_STEPS));

		if( animationStep < ANIMATION_STEPS )
		{
			//for( uint32_t n=0; n < DOF; n++ )
			/*ref[0] += dT[0];
			ref[1] += dT[1];	//ref[4] += dT[1];
			ref[2] += dT[2];	//ref[8] += dT[2];*/

			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MIN + step * float(animationStep);
		}
		else if( animationStep < ANIMATION_STEPS * 2 )
		{
			/*ref[0] -= dT[0];
			ref[1] -= dT[1];
			ref[2] -= dT[2];*/

			animationStep++;
			printf("animation step %u\n", animationStep);

			for( uint32_t n=0; n < DOF; n++ )
				ref[n] = JOINT_MAX - step * float(animationStep-ANIMATION_STEPS);
		}
		else
		{
			animationStep = 0;

			//const float r = float(rand()) / float(RAND_MAX);
			//setAnimationTarget( 10000.0f, 0.0f );
		}

		return true;
	}

	return false;
}


// called by the world update start event
void ArmPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
{
	if( !updateJoints() )
		return;

	printf("%f  %f  %f\n", ref[0], ref[1], ref[2]);

	double angle(1);
	//std::string j2name("joint1");   
	j2_controller->SetJointPosition(this->model->GetJoint("joint1"),  ref[0]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint2"),  ref[1]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint3"),  ref[2]);
	/*j2_controller->SetJointPosition(this->model->GetJoint("joint4"),  ref[3]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint5"),  ref[4]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint6"),  ref[5]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint7"),  ref[6]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint8"),  ref[7]);
	j2_controller->SetJointPosition(this->model->GetJoint("joint9"),  ref[8]);*/
}


// Inverse kinematics solver
void IK( float x, float y, float theta[3] )
{
	const float l1 = 4000.0f;
	const float l2 = 4000.0f;
	const float l3 = 1000.0f;

	const float phi = 0.0f;

	const float xw = x - l3 * cosf(0.0f);
	const float yw = y - l3 * sinf(0.0f);

	const float l12 = l1 * l1;
	const float l22 = l2 * l2;

	const float xw2 = xw * xw;
	const float yw2 = yw * yw;

	if( xw == 0.0f && yw == 0.0f )
	{
		theta[0] = 0.0f;
		theta[1] = 0.0f;
		theta[2] = 0.0f;
	}
	else
	{
		theta[1] = PI - acosf((l12+l22-xw2-yw2)/(2*l1*l2));
		theta[0] = atanf(yw/xw) - acosf((l12-l22+xw2+yw2)/(2*l1*sqrtf(xw2+yw2)));
		theta[2] = phi - theta[1] - theta[0];
	}
}


// setAnimationTarget
void ArmPlugin::setAnimationTarget( float x, float y )
{
	IK( x, y, dT );

	printf("theta:  %f  %f  %f\n", dT[0], dT[1], dT[2]);
	
	dT[0] /= float(ANIMATION_STEPS);
	dT[1] /= float(ANIMATION_STEPS);
	dT[2] /= float(ANIMATION_STEPS);

	printf("dT:  %f  %f  %f\n", dT[0], dT[1], dT[2]);
}

}

