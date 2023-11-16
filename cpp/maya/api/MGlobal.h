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
#include <maya/MFnToolContext.h>

// Iterators

// Proxies
#include <maya/MPxNode.h>

// Custom
#include "MObject.h"




namespace LMGLobal {
	/* LMScene
	 * Lunar Maya Global wrapper class.
	 */

	inline bool currentToolIsTransformContext() {
		/* Checks wheter or not the current tool context is the 'Move Tool' or 'Rotate Tool'

		 */
		MFnToolContext fnCurrentToolContext = MGlobal::currentToolContext();
		MString strToolTitle = fnCurrentToolContext.title();
		if (strToolTitle == "Move Tool" || strToolTitle == "Rotate Tool") {
			return true;
		}

		return false;
	}


}
