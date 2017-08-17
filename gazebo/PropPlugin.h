/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __GAZEBO_PROP_PLUGIN_H__
#define __GAZEBO_PROP_PLUGIN_H__


#include <boost/bind.hpp>
#include <gazebo/gazebo.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>
#include <stdio.h>
#include <iostream>
#include <gazebo/transport/TransportTypes.hh>
#include <gazebo/msgs/MessageTypes.hh>
#include <gazebo/common/Time.hh>


#include <errno.h>
#include <fcntl.h>
#include <assert.h>
#include <unistd.h>
#include <pthread.h>
#include <ctype.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <string.h>
#include <syslog.h>
#include <time.h>



namespace gazebo
{

/**
 * PropPlugin
 */
class PropPlugin : public ModelPlugin
{
public:
	virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/);
	virtual void OnUpdate(const common::UpdateInfo & /*_info*/);

	void ResetDynamics();
	void Randomize();

	physics::ModelPtr model;	// Pointer to the model

private:
	
	// Original pose the prop was created in
	math::Pose originalPose;

	// Pointer to the update event connection
	event::ConnectionPtr updateConnection;
};


/**
 * Retrieve the number of global Prop objects
 */
size_t GetNumProps();

/**
 * Retrieve a global Prop object by index
 */
PropPlugin* GetProp( size_t index );

/**
 * Retrieve a global Prop object by name
 */
PropPlugin* GetPropByName( const char* name );

/**
 * Reset all prop poses and dynamics
 */
void ResetPropDynamics();

/**
 * Randomize prop locations
 */
void RandomizeProps();

}

#endif
