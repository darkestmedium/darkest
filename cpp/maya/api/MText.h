#pragma once

// System Includes
#include <sstream>
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
#include "MathUtility.h"
#include "MObject.h"




namespace LMText {
	/* LMText
	 * Lunar Maya Text wrapper class.
	 */

	inline std::string doublePrecision(const double number, unsigned int decimalDigits) {
		/* Set the decimal precision od doubles.
		
		Args:
			number (double): Given number
		
		Returns:
			decimalDigits (unsigned int): Decimal precision

		*/
		std::stringstream ss;
		ss.precision(decimalDigits);  // set # places after decimal
		ss << std::fixed;
		ss << number;

		return ss.str();
	}

}
