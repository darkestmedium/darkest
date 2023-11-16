#pragma once

// System Includes
// #include <cmath>
#include <string.h>
#include <cassert>
#include <map>
#include <vector>
#include <set>

// Maya General Includes
#include <maya/MGlobal.h>
#include <maya/MAnimControl.h>
#include <maya/MDagPath.h>
#include <maya/MDagPathArray.h>
#include <maya/MMatrix.h>
#include <maya/MAngle.h>
#include <maya/MQuaternion.h>
#include <maya/MEulerRotation.h>
#include <maya/MPoint.h>
#include <maya/MVector.h>
#include <maya/MString.h>
#include <maya/MDataHandle.h>
#include <maya/MSelectionList.h>

// Function Sets
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnTransform.h>
#include <maya/MFnToolContext.h>

// Iterators

// Proxies
#include <maya/MPxNode.h>

// Lunar
#include "../maya/api/MAttribute.h"
#include "../maya/api/MGlobal.h"
#include "../maya/api/MAnimControl.h"
#include "../maya/api/MSolve.h"
#include "../maya/api/MRigUtils.h"
#include "../maya/api/MPlugin.h"
#include "../maya/api/Utils.h"
#include "../maya/api/MathUtility.h"




class Ik2bSolver : public MPxNode {
public:
	// Public Data
	static const MString typeName;
	static const MTypeId typeId;

	// Node's Input Attributes
	static MObject attrInFkIk;
	static Attribute attrInFkStart, attrInFkMid, attrInFkEnd;
	static Attribute attrInIkStart, attrInIkMid, attrInIkEnd;
	static Attribute attrInIkHandle;
	static MObject attrInPvX, attrInPvY, attrInPvZ, attrInPv;
	static Attribute attrInTwist;
	static MObject attrInSoftness;
	static Attribute attrInOutStart, attrInOutMid, attrInOutEnd;

	// Nodes's Output Attributes
	static MObject attrOutStartX, attrOutStartY, attrOutStartZ, attrOutStart;
	static MObject attrOutMidX, attrOutMidY, attrOutMidZ, attrOutMid;
	static MObject attrOutEndX, attrOutEndY, attrOutEndZ, attrOutEnd;
	static MObject attrOutFkVisibility, attrOutIkVisibility;
	static MObject attrOutUpdate;

	// static MObject attrInTime;

	// In data
	MMatrix matInFkStart, matInFkMid, matInFkEnd;
	MMatrix matInIkStart, matInIkMid, matInIkEnd;
	MMatrix matInIkHandle;
	MVector posInPv;
	MAngle twist;
	double fkIk, softness;
	bool bIsPvConnected, bFkVisibility, bIkVisibility;
	MAngle::Unit uiUnitAngle;

	MEulerRotation eulFkStart, eulFkMid, eulFkEnd;

	// Function sets
	MFnTransform fnRoot, fnFkStart, fnFkMid, fnFkEnd;
	MFnTransform fnIkStart, fnIkMid, fnIkEnd, fnIkHandle, fnPv;
	MFnTransform fnOutStart, fnOutMid, fnOutEnd;

	// Position
	MVector posIkRoot, posIkStart, posIkMid, posIkEnd, posIkHandle, posIkPv;

	// Quats
	MQuaternion quatFkStart, quatFkMid, quatFkEnd;
	MQuaternion quatIkStart, quatIkMid, quatIkEnd;
	MQuaternion quatOutStart, quatOutMid, quatOutEnd;
	
	MObject objSelf;

	// Helpers
	MSelectionList listSel;
	MAnimControl ctrlAnim;

	// Constructors
	Ik2bSolver()
		: MPxNode()
		, fkIk(0.0)
		, bFkVisibility(true)
		, bIkVisibility(false)
		, bIsPvConnected(false)
		, uiUnitAngle(MAngle::uiUnit())
	{};
	// Destructors
	~Ik2bSolver() override {};

	// Public methods - overrides
	static void* creator() {return new Ik2bSolver();}
	static MStatus initialize();
	virtual MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;
	MStatus setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) override;
	void getCacheSetup(const MEvaluationNode& evalNode,	MNodeCacheDisablingInfo& disablingInfo,	MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const override;
	void postConstructor() override;
	SchedulingType schedulingType() const override {return SchedulingType::kParallel;}

	// Custom solver methods
	void getFkTransforms();
	void getIkTransforms();

	MStatus solveLimb();
	bool solveFk();
	bool solveIk();
	void blendFkIk();
	void solveFkIk();

	MStatus parseDataBlock(MDataBlock& dataBlock);
	MStatus updateOutput(const MPlug& plug, MDataBlock& dataBlock);
};