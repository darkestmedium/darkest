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




namespace MScene {
	/* MScene
	 * Lunar Maya Scene wrapper class.
	 */
	inline MObject getTime1Node() {return MObject::get_obj_from_string("time1");}

	inline MStatus connectSceneTime(MObject& object, MString plug, MDGModifier& modDg) {
		/* Connects the scene's default time1 node to the given target.
		*/
		MFnDependencyNode fnDestinationNode = object;
		MPlug plugDestinationInTime = fnDestinationNode.findPlug(plug, false);

		MFnDependencyNode fnTimeNode = getTime1Node();
		MPlug plugTimeOutTime = fnTimeNode.findPlug("outTime", false);
		
		// MDGModifier dgMod;
		modDg.connect(plugTimeOutTime, plugDestinationInTime);
		// dgMod.doIt();
	
		return MS::kSuccess;
	};

}
