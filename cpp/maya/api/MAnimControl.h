#pragma once

#include "MathUtility.h"

// System Includes
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

// Maya General Includes
#include <maya/MAnimControl.h>
#include <maya/MTime.h>
#include <maya/MSelectionList.h>
#include <maya/MAngle.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MEulerRotation.h>
#include <maya/MGlobal.h>
#include <maya/MMatrix.h>
#include <maya/MVector.h>
#include <maya/MQuaternion.h>
#include <maya/MDGModifier.h>
#include <maya/MDagPath.h>

// Function Sets
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnUnitAttribute.h>
// #include <maya/MFnPlugin.h>

// Iterators

// Proxies
#include <maya/MPxNode.h>

// Custom
#include "MObject.h"




namespace DAnimControl {
	/* LMAnimControl
	 * Lunar Maya AnimControl	wrapper with additional methods.
	 */


	inline bool timeChanged(MAnimControl& animCtrl, MTime& timeCached, MTime& timeCurrent) {
		/* Checks wheter or not time has changed.

		Playback / scrubbing / time change
		if (AnimCtrl.isPlaying() or AnimCtrl.isScrubbing() or TimeCached != TimeCurrent)

		Args:
			animCtrl (MAnimControl&): Animation control instance to check isPlaying and isSrubbing.
			timeCached (MTimge&): Time that has been cached - ex. previous frame.
			timeCurrent (MTimge&): Current time / frame.

		Return:
			bool: True if time has changed, false if it didn't.

		*/
		if (animCtrl.isPlaying() || animCtrl.isScrubbing() || timeCached != timeCurrent) {
			return true;
		}

		return false;
	};


	inline MObject get_time1_node() {return MObject::get_obj_from_string("time1");}

	inline MStatus connect_scene_time(MObject& object, MDGModifier& mod_dg, MString plug="inTime") {
		/* Connects the scene's default time1 node to the given target.
		*/
		MFnDependencyNode fnDestinationNode = object;
		MPlug plug_in_time = fnDestinationNode.findPlug(plug, false);

		MFnDependencyNode fnTimeNode = get_time1_node();
		MPlug plug_out_time = fnTimeNode.findPlug("outTime", false);
		
		// MDGModifier mod_dg;
		mod_dg.connect(plug_in_time, plug_out_time);
		mod_dg.doIt();
	
		return MS::kSuccess;
	};

}
