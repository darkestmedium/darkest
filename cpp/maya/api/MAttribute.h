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
#include "MScene.h"




namespace MAttribute {
	/* LMAttrUtilis
	 * Wrapper namespace for attribute utilities.
	 */


	inline MStatus lockAndHideAttr(MPlug& plug) {
		/* Locks and hides the given plug from the channelbox.

		Returns:
			status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
				during the command.

		*/
		MStatus status;
		plug.setLocked(true);
		plug.setKeyable(false);
		plug.setChannelBox(false);

		return MS::kSuccess;
	};


	inline MStatus lockAndHideAttr(MObject& node, MObject& attribute) {
		/* Locks and hides the given plug from the channelbox.

		Returns:
			status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
				during the command.

		*/
		MStatus status;
		MPlug plug(node, attribute);
		plug.setLocked(true);
		plug.setKeyable(false);
		plug.setChannelBox(false);

		return MS::kSuccess;
	};


	inline MObject getSourceObjFromPlug(const MObject& object, const MObject& plug) {
		/* Gets the object from the given plug if it is connected.

		Args:
			Plug (MObject&): Object for the given plug.
		
		Returns:
			MObject: If the plug is a valid connection it will return the obj, otherwise a null object will
				be returned instead.

		*/
		MPlug plugDestination(object, plug);
		if (plugDestination.isConnected()) {return plugDestination.source().node();}

		return MObject::kNullObj;
	};

}
