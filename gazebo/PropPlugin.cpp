/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "PropPlugin.h"


namespace gazebo
{

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(PropPlugin)

//---------------------------------------------------------------------------------------
std::vector<PropPlugin*> props;

size_t GetNumProps()
{
	return props.size();
}

PropPlugin* GetProp( size_t index )
{
	return props[index];
}

PropPlugin* GetPropByName( const char* name )
{
	if( !name )
		return NULL;

	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
	{
		if( strcmp(props[n]->model->GetName().c_str(), name) == 0 )
			return props[n];
	}

	return NULL;
}

void ResetPropDynamics()
{
	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
		props[n]->ResetDynamics();
}

void RandomizeProps()
{
	const size_t numProps = props.size();

	for( size_t n=0; n < numProps; n++ )
		props[n]->Randomize();
}

//---------------------------------------------------------------------------------------


// Plugin init
void PropPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
{
	printf("PropPlugin::Load('%s')\n", _parent->GetName().c_str());

	// Store the pointer to the model
	this->model = _parent;

	// Store the original pose of the model
	this->originalPose = model->GetWorldPose();

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&PropPlugin::OnUpdate, this, _1));

	// Track this object in the global Prop registry
	props.push_back(this);
}


// Called by the world update start event
void PropPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
{
	// Apply a small linear velocity to the model.
	//this->model->SetLinearVel(math::Vector3(.03, 0, 0));

   /*const math::Pose& pose = model->GetWorldPose();
	
	printf("%s location:  %lf %lf %lf\n", model->GetName().c_str(), pose.pos.x, pose.pos.y, pose.pos.z);
	
	const math::Box& bbox = model->GetBoundingBox();

	printf("%s bounding:  min=%lf %lf %lf  max=%lf %lf %lf\n", model->GetName().c_str(), bbox.min.x, bbox.min.y, bbox.min.z, bbox.max.x, bbox.max.y, bbox.max.z);
   */
	/*const math::Box& bbox = model->GetBoundingBox();

  	const math::Vector3 center = bbox.GetCenter();
	const math::Vector3 bbSize = bbox.GetSize();

	printf("%s bounding:  center=%lf %lf %lf  size=%lf %lf %lf\n", model->GetName().c_str(), center.x, center.y, center.z, bbSize.x, bbSize.y, bbSize.z); 
	*/
}


// Reset the model's dynamics and pose to original
void PropPlugin::ResetDynamics()
{
	model->SetAngularAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetAngularVel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearVel(math::Vector3(0.0, 0.0, 0.0));

	model->SetWorldPose(originalPose);
}


inline float randf( float rand_min, float rand_max )
{
	const float r = float(rand()) / float(RAND_MAX);
	return (r * (rand_max - rand_min)) + rand_min;
}


// Randomize
void PropPlugin::Randomize()
{
	model->SetAngularAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetAngularVel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearAccel(math::Vector3(0.0, 0.0, 0.0));
	model->SetLinearVel(math::Vector3(0.0, 0.0, 0.0));

	math::Pose pose = originalPose;

	pose.pos.x = randf(0.35f, 0.45f);
	pose.pos.y = randf(-1.5f, 0.2f);
	pose.pos.z = 0.0f;
	
	printf("prop random pos:  %f  %f  %f\n", pose.pos.x, pose.pos.y, pose.pos.z);
	model->SetWorldPose(pose);
}

}

