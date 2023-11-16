#pragma once

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
#include "Utils.h"
#include "MathUtility.h"
#include "MObject.h"




namespace LMSolve {
	/* LMSolver
	 * Lunar Maya Solver utiliteis
	 */


	inline double softenIk(double lenAT, double lenAB, double vecBC, double lenABC, double softness) {
		/* Softness the AT length if required.
		*/
		lenAT = std::max(lenAT, lenAB - vecBC);
		double da = lenABC - softness;
		double softAT = da + softness * (1.0 - std::exp((da-lenAT)/softness));
		return (lenAT > da && da > 0.0) ? softAT : lenAT;
	}


	inline MStatus twoBoneIk(
		const MVector& vecA, const MVector& vecB, const MVector& vecC, const MVector& vecT, const MVector& vecPv,
		MAngle& twist, double softness, bool bIsPvConnected,
		MQuaternion& quatA, MQuaternion& quatB
		) {
		/* Calculates the ik for a two bone limb.
		
		Reference:
			https://theorangeduck.com/page/simple-two-joint
			https://github.com/chadmv/cmt/blob/master/src/ikRigNode.cpp

		*/
		// MStatus status;

		// From to Vectors - reusable
		MVector vecAB = vecB - vecA;
		MVector vecAC = vecC - vecA;
		MVector vecAT = vecT - vecA;
		MVector vecBC = vecC - vecB;
		MVector vecAPv = vecPv - vecA;
	
		// Direction vector
		MVector vecD = (vecB - (vecA + (vecAC * (vecAB * vecAC)))).normal();
	
		// Lengths
		double lenAB = vecAB.length();
		double lenCB = (vecB - vecC).length();
		double lenABC = lenAB + lenCB;
		double lenAT = clamp(vecAT.length(), kEpsilon, lenABC - kEpsilon);

		// Soften the edge if required
		if (softness > 0.0) {lenAT = softenIk(vecAT.length(), lenAB, vecBC.length(), lenABC, softness);}

		// Get current interior angles of start and mid
		double ac_ab_0 = acos(clamp((vecAC).normal() * (vecAB).normal(), -1.0, 1.0));
		double ba_bc_0 = acos(clamp((vecA - vecB).normal() * vecBC.normal(), -1.0, 1.0));
		double ac_at_0 = acos(clamp((vecAC).normal() * (vecAT).normal(), -1.0, 1.0));
		// Get desired interior angles
		double ac_ab_1 = acos(clamp((lenCB * lenCB - lenAB * lenAB - lenAT * lenAT) / (-2 * lenAB * lenAT), -1.0, 1.0));
		double ba_bc_1 = acos(clamp((lenAT * lenAT - lenAB * lenAB - lenCB * lenCB) / (-2 * lenAB * lenCB), -1.0, 1.0));

		MVector axis0 = (vecAC ^ vecD).normal();
		MVector axis1 = (vecAC ^ vecAT).normal();

		MQuaternion r0(ac_ab_1 - ac_ab_0, axis0);
		MQuaternion r1(ba_bc_1 - ba_bc_0, axis0);
		MQuaternion r2(ac_at_0, axis1);

		// Pole vector rotation
		// Determine the rotation used to rotate the normal of the triangle formed by
		// a.b.c post r0*r2 rotation to the normal of the triangle formed by triangle a.pv.t
		MVector n1 = (vecAC ^ vecAB).normal().rotateBy(r0).rotateBy(r2);
		MVector n2 = (vecAT ^ vecAPv).normal();
		MQuaternion r3 = n1.rotateTo(n2);

		// Rotation cross vectors and twist
		MQuaternion rT(twist.asRadians(), vecAT);

		quatA *= r0 * r2 * r3 * rT;

		// local rotation because we output to the skeleton
		// Since we are calculating in world space, apply the start rotations to the mid
		quatB *= r1;
		quatB *= r0 * r2 * r3 * rT;

		return MS::kSuccess;
	}
};