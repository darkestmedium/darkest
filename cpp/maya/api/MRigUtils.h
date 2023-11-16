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




namespace LMRigUtils {
	/* LMRigUtils
	 * Lunar Maya Rig Utilities
	 */

	inline MVector getPvPosition(MVector& vecA, MVector& vecB, MVector& vecC, MString space="world") {
		/* Gets the pole vector position from the input of three vectors.

		From Greg's Hendrix tutorial https://www.youtube.com/watch?v=bB_HL1tBVHY

		*/
		MVector vecAC = vecC - vecA;
		MVector vecAB = vecB - vecA;
		MVector vecBC = vecC - vecB;

		double valScale = (vecAC * vecAB) / (vecAC * vecAC);
		MVector vecProjection = vecAC * valScale + vecA;
		double lenABC = vecAB.length() + vecBC.length();

		MVector posPv((vecB - vecProjection).normal() * lenABC);

		if (space == "world") {return posPv + vecB;}
		return posPv;
	}
};

