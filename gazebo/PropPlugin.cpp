/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#include "PropPlugin.h"


namespace gazebo
{

// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(PropPlugin)



// Plugin init
void PropPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
{
	// Store the pointer to the model
	this->model = _parent;

	// Listen to the update event. This event is broadcast every simulation iteration.
	this->updateConnection = event::Events::ConnectWorldUpdateBegin(boost::bind(&PropPlugin::OnUpdate, this, _1));
}


// Called by the world update start event
void PropPlugin::OnUpdate(const common::UpdateInfo & /*_info*/)
{
	// Apply a small linear velocity to the model.
	this->model->SetLinearVel(math::Vector3(.03, 0, 0));
}


}
