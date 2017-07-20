/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "ArmPlugin.h"


#define PI 3.141592653589793238462643383279502884197169f

#define ANIMATION_STEPS 2000
#define JOINT_MIN	-0.75f
#define JOINT_MAX	 0.75f


#define COLLISION_FILTER "ground_plane::link::collision"


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



namespace gazebo
{
 
// register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(ArmPlugin)


// constructor
ArmPlugin::ArmPlugin() : ModelPlugin(), cameraNode(new gazebo::transport::Node()), collisionNode(new gazebo::transport::Node())
{
	for( uint32_t n=0; n < DOF; n++ )
		ref[n] = JOINT_MIN; //0.0f;

	printf("HELLO WORLD!\n");

	animationStep = 0;
	agent = NULL;

	//setAnimationTarget(7000, 5000);
}


// Load
void ArmPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) 
{
	printf("ArmPlugin::Load()\n");

	// Create AI agent
	//agent = dqnAgent::Create(40, 80, 3, DOF*2+1);

	//if( !agent )
	//	printf("ArmPlugin - failed to create AI agent\n");

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


// onCameraMsg
void ArmPlugin::onCameraMsg(ConstImageStampedPtr &_msg)
{
	// Dump the message contents to stdout.
	printf("camera callback\n");

	if( !_msg )
	{
		printf("NULL message\n");
		return;
	}

	const int width  = _msg->image().width();
	const int height = _msg->image().height();
	const int bpp    = (_msg->image().step() / _msg->image().width()) * 8;	// bits per pixel
	const int size   = _msg->image().data().size();
	//const int width = _msg->width();
	//const int height = _msg->height();

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
	printf("collision callback\n");

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


// called by the world update start event
void ArmPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
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

}

