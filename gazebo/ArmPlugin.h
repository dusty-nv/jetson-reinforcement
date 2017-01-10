/*
 * http://github.com/dusty-nv/jetson-reinforcement
 */

#ifndef __GAZEBO_ARM_PLUGIN_H__
#define __GAZEBO_ARM_PLUGIN_H__


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

class ArmPlugin : public ModelPlugin
{
public: 

	ArmPlugin(); 

	virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/); 
	virtual void OnUpdate(const common::UpdateInfo & /*_info*/);

	void setAnimationTarget( float x, float y );

	void onCameraMsg(ConstImageStampedPtr &_msg);
	void onCollisionMsg(ConstContactsPtr &contacts);

	static const uint32_t DOF = 3;

private:
	physics::ModelPtr model;	 // Pointer to the model

	// Pointer to the update event connection
	event::ConnectionPtr updateConnection;

	physics::JointController * j2_controller;
	//enum ach_status ach_close(ach_channel_t *channel);
	//enum ach_status ach_unlink(const char *name);

	float ref[DOF];
	float dT[3];
	uint32_t animationStep;

	gazebo::transport::NodePtr cameraNode;
	gazebo::transport::SubscriberPtr cameraSub;

	gazebo::transport::NodePtr collisionNode;
	gazebo::transport::SubscriberPtr collisionSub;
};

}


#endif
