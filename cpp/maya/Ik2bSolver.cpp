#include "Ik2bSolver.h"


// Public Data
const MString Ik2bSolver::typeName = "ik2bSolver";
const MTypeId Ik2bSolver::typeId = 0x9000020;

// Node's Input Attributes
MObject Ik2bSolver::attrInFkIk;

Attribute Ik2bSolver::attrInFkStart, Ik2bSolver::attrInFkMid, Ik2bSolver::attrInFkEnd;
Attribute Ik2bSolver::attrInIkStart, Ik2bSolver::attrInIkMid, Ik2bSolver::attrInIkEnd;
Attribute Ik2bSolver::attrInIkHandle;

MObject Ik2bSolver::attrInPvX, Ik2bSolver::attrInPvY, Ik2bSolver::attrInPvZ, Ik2bSolver::attrInPv;

Attribute Ik2bSolver::attrInTwist;
MObject Ik2bSolver::attrInSoftness;

// Nodes's Output Attributes
MObject Ik2bSolver::attrOutStartX, Ik2bSolver::attrOutStartY, Ik2bSolver::attrOutStartZ, Ik2bSolver::attrOutStart;
MObject Ik2bSolver::attrOutMidX, Ik2bSolver::attrOutMidY, Ik2bSolver::attrOutMidZ, Ik2bSolver::attrOutMid;
MObject Ik2bSolver::attrOutEndX, Ik2bSolver::attrOutEndY, Ik2bSolver::attrOutEndZ, Ik2bSolver::attrOutEnd;

MObject Ik2bSolver::attrOutFkVisibility;
MObject Ik2bSolver::attrOutIkVisibility;

Attribute Ik2bSolver::attrInOutStart;
Attribute Ik2bSolver::attrInOutMid;
Attribute Ik2bSolver::attrInOutEnd;

MObject Ik2bSolver::attrOutUpdate;




MStatus Ik2bSolver::initialize() {
	/* Node Initializer.

	This method initializes the node, and should be overridden in user-defined nodes.
	
	Returns:
		status code (MStatus): kSuccess if the operation was successful, kFailure if an	error occured
			during the operation.

	*/
	MStatus status;

	// Attr function sets
	MFnNumericAttribute nAttr;
	MFnMatrixAttribute mAttr;
	MFnUnitAttribute uAttr;
	MFnEnumAttribute eAttr;

	// Node's Input Attributes
	attrInFkIk = nAttr.create("fkIk", "fkIk", MFnNumericData::kDouble);
	nAttr.setKeyable(true);
	nAttr.setReadable(false);
	nAttr.setStorable(true);
	nAttr.setMin(0.0);
	nAttr.setMax(100.0);

	createAttribute(attrInFkStart, "fkStart", DefaultValue<MMatrix>());
	createAttribute(attrInFkMid, "fkMid", DefaultValue<MMatrix>());
	createAttribute(attrInFkEnd, "fkEnd", DefaultValue<MMatrix>());

	createAttribute(attrInIkStart, "ikStart", DefaultValue<MMatrix>());
	createAttribute(attrInIkMid, "ikMid", DefaultValue<MMatrix>());
	createAttribute(attrInIkEnd, "ikEnd", DefaultValue<MMatrix>());
	createAttribute(attrInIkHandle, "ikHandle", DefaultValue<MMatrix>());

	createAttribute(attrInOutStart, "outStart", DefaultValue<MMatrix>());
	createAttribute(attrInOutMid, "outMid", DefaultValue<MMatrix>());
	createAttribute(attrInOutEnd, "outEnd", DefaultValue<MMatrix>());

	attrInPvX = nAttr.create("poleVectorX", "pvX", MFnNumericData::kDouble, 0.0);
	attrInPvY = nAttr.create("poleVectorY", "pvY", MFnNumericData::kDouble, 0.0);
	attrInPvZ = nAttr.create("poleVectorZ", "pvZ", MFnNumericData::kDouble, 0.0);
	attrInPv = nAttr.create("poleVector", "pv", attrInPvX, attrInPvY, attrInPvZ);
	nAttr.setReadable(false);

	createAttribute(attrInTwist, "twist", DefaultValue<double>());

	attrInSoftness = nAttr.create("softness", "sfns", MFnNumericData::kDouble, 0.0);
	nAttr.setKeyable(true);
	nAttr.setReadable(false);
	nAttr.setStorable(true);
	nAttr.setWritable(true);
	nAttr.setMin(0.0);
	nAttr.setMax(10.0);

	// Output attributes
	attrOutStartX = uAttr.create("rotateStartX", "rsX", MFnUnitAttribute::kAngle, 0.0);
	attrOutStartY = uAttr.create("rotateStartY", "rsY", MFnUnitAttribute::kAngle, 0.0);
	attrOutStartZ = uAttr.create("rotateStartZ", "rsZ", MFnUnitAttribute::kAngle, 0.0);
	attrOutStart = nAttr.create("rotateStart", "rs", attrOutStartX, attrOutStartY, attrOutStartZ);
	nAttr.setWritable(false);

	attrOutMidX = uAttr.create("rotateMidX", "rmX", MFnUnitAttribute::kAngle, 0.0);
	attrOutMidY = uAttr.create("rotateMidY", "rmY", MFnUnitAttribute::kAngle, 0.0);
	attrOutMidZ = uAttr.create("rotateMidZ", "rmZ", MFnUnitAttribute::kAngle, 0.0);
	attrOutMid = nAttr.create("rotateMid", "rm", attrOutMidX, attrOutMidY, attrOutMidZ);
	nAttr.setWritable(false);

	attrOutEndX = uAttr.create("rotateEndX", "reX", MFnUnitAttribute::kAngle, 0.0);
	attrOutEndY = uAttr.create("rotateEndY", "reY", MFnUnitAttribute::kAngle, 0.0);
	attrOutEndZ = uAttr.create("rotateEndZ", "reZ", MFnUnitAttribute::kAngle, 0.0);
	attrOutEnd = nAttr.create("rotateEnd", "re", attrOutEndX, attrOutEndY, attrOutEndZ);
	nAttr.setWritable(false);

	attrOutFkVisibility = nAttr.create("fkVisibility", "fkVis", MFnNumericData::kBoolean, true);
	nAttr.setWritable(false);

	attrOutIkVisibility = nAttr.create("ikVisibility", "ikVis", MFnNumericData::kBoolean, false);
	nAttr.setWritable(false);

	attrOutUpdate = nAttr.create("update", "upt", MFnNumericData::kDouble, 0.0);
	nAttr.setWritable(false);

	// Add attributes
	addAttributes(
		attrInFkStart, attrInFkMid,	attrInFkEnd,
		attrInIkStart, attrInIkMid,	attrInIkEnd,
		attrInIkHandle, attrInPv,
		attrInFkIk, attrInTwist, attrInSoftness,
		attrInOutStart, attrInOutMid, attrInOutEnd,
		attrOutStart, attrOutMid, attrOutEnd,
		attrOutFkVisibility, attrOutIkVisibility,
		attrOutUpdate
	);

	return MS::kSuccess;
}


MStatus Ik2bSolver::parseDataBlock(MDataBlock& dataBlock) {
	/* Parse the data block and get all inputs.
	
	We're getting the mObj from the .attribute() instead of a numeric data type like double in
	order to retrieve the MFnTransform for the input controllers - this also triggers the input as
	dirty. All of Maya's solvers get the world position from the .rotatePivot() method.

	*/
	MStatus status;

	fkIk = dataBlock.inputValue(attrInFkIk).asDouble();
	if (fkIk == 0.0) {
		bFkVisibility = true;
		bIkVisibility = false;
	}	else if (fkIk > 0.0 && fkIk < 100.0) {
		bFkVisibility = true;
		bIkVisibility = true;
	} else if (fkIk == 100.0) {
		bFkVisibility = false;
		bIkVisibility = true;
	}

	// Ask for time value to force refresh on the node
	// timeCurrent = dataBlock.inputValue(attrInTime, &status).asTime();
	// Asking for the actuall matrix input helps refreshing the rig if there are no anim curves
	matInFkStart = dataBlock.inputValue(attrInFkStart).asMatrix();
	matInFkMid = dataBlock.inputValue(attrInFkMid).asMatrix();
	matInFkEnd = dataBlock.inputValue(attrInFkEnd).asMatrix();

	matInIkStart = dataBlock.inputValue(attrInIkStart).asMatrix();
	matInIkMid = dataBlock.inputValue(attrInIkMid).asMatrix();
	matInIkEnd = dataBlock.inputValue(attrInIkEnd).asMatrix();

	matInIkHandle = dataBlock.inputValue(attrInIkHandle).asMatrix();
	posInPv = MVector(dataBlock.inputValue(attrInPvX).asDouble(), dataBlock.inputValue(attrInPvY).asDouble(),	dataBlock.inputValue(attrInPvZ).asDouble());

	// In controllers Fk / Ik
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnFkStart, attrInFkStart));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnFkMid, attrInFkMid));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnFkEnd, attrInFkEnd));

	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnIkStart, attrInIkStart));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnIkMid, attrInIkMid));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnIkEnd, attrInIkEnd));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnIkHandle, attrInIkHandle));

	// Out controllers
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnOutStart, attrInOutStart));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnOutMid, attrInOutMid));
	CHECK_MSTATUS_AND_RETURN_IT(LMPlugin::parseTransformInput(dataBlock, objSelf, fnOutEnd, attrInOutEnd));

	// Pole vector
	MDagPath pathPv;
	status = MDagPath::getAPathTo(MAttribute::getSourceObjFromPlug(objSelf, dataBlock.inputValue(attrInPv).attribute()), pathPv);
	if (status == MS::kSuccess) {
		fnPv.setObject(pathPv);
		bIsPvConnected = true;
	} else {
		fnPv.setObject(MObject::kNullObj);
		bIsPvConnected = false;
		// Get fk start parent
		MDagPath pathRoot;
		status = MDagPath::getAPathTo(fnFkStart.parent(0), pathRoot);
		if (status == MS::kSuccess) {
			fnRoot.setObject(pathRoot);
		} else {
			fnRoot.setObject(MObject::kNullObj);
		}
	}

	// Additional attributes
	uiUnitAngle = MAngle::uiUnit();
	twist = MAngle(dataBlock.inputValue(attrInTwist).asDouble(), uiUnitAngle);
	softness = dataBlock.inputValue(attrInSoftness).asDouble();

	return MS::kSuccess;
}


void Ik2bSolver::getFkTransforms() {
	/* Used in fk/ik and ik - needs world space
	*/
	fnFkStart.getRotation(quatFkStart, MSpace::kWorld);
	fnFkMid.getRotation(quatFkMid, MSpace::kWorld);
	fnFkEnd.getRotation(quatFkEnd, MSpace::kWorld);
}


void Ik2bSolver::getIkTransforms() {
	/* Used in fk / ik and ik - needs world space
	*/
	posIkStart = fnIkStart.rotatePivot(MSpace::kWorld);
	fnIkStart.getRotation(quatIkStart, MSpace::kWorld);

	posIkMid = fnIkMid.rotatePivot(MSpace::kWorld);
	fnIkMid.getRotation(quatIkMid, MSpace::kWorld);

	posIkEnd = fnIkEnd.rotatePivot(MSpace::kWorld);

	posIkHandle = fnIkHandle.rotatePivot(MSpace::kWorld);
	fnIkHandle.getRotation(quatIkEnd, MSpace::kWorld);

	if (bIsPvConnected) {posIkPv = fnPv.rotatePivot(MSpace::kWorld);}
	else {
		// we need a getPvRootPosition that will calculate the pv and multiply by the root
		posIkRoot = fnRoot.rotatePivot(MSpace::kWorld);
		posIkPv = posInPv * fnRoot.dagPath().exclusiveMatrix() + posIkRoot;
	}
}


MStatus Ik2bSolver::solveLimb() {
	/* Solves the limb. 

	Main fk / ik routing method. 

	*/
	if (fkIk == 0) {
		solveFk();
	} else if (fkIk > 0.0 && fkIk < 100.0) {
		solveFkIk();
	} else if (fkIk == 100.0) {
		solveIk();
	}

	return MS::kSuccess;
}


bool Ik2bSolver::solveFk() {
	/* Set the fk transforms.

	We don't actually solve fk - it's called like this just for consistency and readability.
	The isEditing flag is reserved for editing the fk transforms where we move the pole vector by
	a constant distance (limb length) calculated from the mid transform. 

	*/
	// Get fk rotations directly in local space since we don't blend here.
	fnFkStart.getRotation(quatOutStart, MSpace::kTransform);
	fnFkMid.getRotation(quatOutMid, MSpace::kTransform);
	fnFkEnd.getRotation(quatOutEnd, MSpace::kTransform);

	return true;
}


bool Ik2bSolver::solveIk() {
	/* Calculates the ik solution for a two bone limb.
	*/
	getIkTransforms();

	LMSolve::twoBoneIk(posIkStart, posIkMid, posIkEnd, posIkHandle, posIkPv, twist, softness, bIsPvConnected, quatIkStart, quatIkMid);

	// Sync the ik chain
	fnOutStart.setRotation(quatIkStart, MSpace::kWorld);
	fnOutMid.setRotation(quatIkMid, MSpace::kWorld);
	fnOutEnd.setRotation(quatIkEnd, MSpace::kWorld);
	// Get the out quats in local space
	fnOutStart.getRotation(quatOutStart, MSpace::kTransform);
	fnOutMid.getRotation(quatOutMid, MSpace::kTransform);
	fnOutEnd.getRotation(quatOutEnd, MSpace::kTransform);
	// Sync the ik chain
	fnIkStart.setRotation(quatIkStart, MSpace::kWorld);
	fnIkMid.setRotation(quatIkMid, MSpace::kWorld);
	fnIkEnd.setRotation(quatIkEnd, MSpace::kWorld);

	return true;
}


void Ik2bSolver::blendFkIk() {
	// because we want to use 0 - 100 in the channel box, yeah i know :|
	double scaledWeight = fkIk * 0.01;

	quatOutStart = slerp(quatFkStart, quatIkStart, scaledWeight);
	quatOutMid = slerp(quatFkMid, quatIkMid, scaledWeight);
	quatOutEnd = slerp(quatFkEnd, quatIkEnd, scaledWeight);
}


void Ik2bSolver::solveFkIk() {
	/* So kind of does what the name says but not really.
	*/
	getFkTransforms();
	getIkTransforms();

	LMSolve::twoBoneIk(posIkStart, posIkMid, posIkEnd, posIkHandle, posIkPv, twist, softness, bIsPvConnected, quatIkStart, quatIkMid);

	// Get the ik handle rotation applied to end ik ctrl (converted from different spaces in rig hierarchy)
	fnOutEnd.setRotation(quatIkEnd, MSpace::kWorld);
	fnOutEnd.getRotation(quatIkEnd, MSpace::kWorld);

	// Need to blend in world space
	blendFkIk();
	// Apply the blended rotations to out ctrls
	fnOutStart.setRotation(quatOutStart, MSpace::kWorld);
	fnOutMid.setRotation(quatOutMid, MSpace::kWorld);
	fnOutEnd.setRotation(quatOutEnd, MSpace::kWorld);
	// Get rot in local space to apply them to the output converted to eulers
	fnOutStart.getRotation(quatOutStart, MSpace::kTransform);
	fnOutMid.getRotation(quatOutMid, MSpace::kTransform);
	fnOutEnd.getRotation(quatOutEnd, MSpace::kTransform);
	// Sync the ik chain
	fnIkStart.setRotation(quatIkStart, MSpace::kWorld);
	fnIkMid.setRotation(quatIkMid, MSpace::kWorld);
	fnIkEnd.setRotation(quatIkEnd, MSpace::kWorld);
}


MStatus Ik2bSolver::updateOutput(const MPlug& plug, MDataBlock& dataBlock) {	
	/* Sets the outputs and data block clean.

	Args:
		plug (MPlug&): Plug representing the attribute that needs to be recomputed.
		dataBlock (MDataBlock&): Data block containing storage for the node's attributes.

	Returns:
		status code (MStatus): kSuccess if the operation was successful, kFailure if an	error occured
			during the operation.

	*/
	MStatus status;

	// Rotation outputs
	MEulerRotation eulrStart = quatOutStart.asEulerRotation();
	MDataHandle dhOutStart = dataBlock.outputValue(attrOutStart, &status);
	dhOutStart.set3Double(eulrStart.x, eulrStart.y, eulrStart.z);
	dhOutStart.setClean();

	MEulerRotation eulrMid = quatOutMid.asEulerRotation();
	MDataHandle dhOutMid = dataBlock.outputValue(attrOutMid, &status);
	dhOutMid.set3Double(eulrMid.x, eulrMid.y, eulrMid.z);
	dhOutMid.setClean();

	MEulerRotation eulrEnd = quatOutEnd.asEulerRotation();
	MDataHandle dhOutEnd = dataBlock.outputValue(attrOutEnd, &status);
	dhOutEnd.set3Double(eulrEnd.x, eulrEnd.y, eulrEnd.z);
	dhOutEnd.setClean();

	// Visibility outputs
	MDataHandle dhOutFkVisibility = dataBlock.outputValue(attrOutFkVisibility, &status);
	dhOutFkVisibility.setBool(bFkVisibility);
	dhOutFkVisibility.setClean();

	MDataHandle dhOutIkVisibility = dataBlock.outputValue(attrOutIkVisibility, &status);
	dhOutIkVisibility.setBool(bIkVisibility);
	dhOutIkVisibility.setClean();

	MDataHandle dhOutUpdate = dataBlock.outputValue(attrOutUpdate, &status);
	dhOutUpdate.setDouble(0.0);
	dhOutUpdate.setClean();

	dataBlock.setClean(plug);

	return MS::kSuccess;
}


MStatus Ik2bSolver::compute(const MPlug& plug, MDataBlock& dataBlock) {
	/* This method should be overridden in user defined nodes.

	Recompute the given output based on the nodes inputs. The plug represents the data
	value that needs to be recomputed, and the data block holds the storage for all of
	the node's attributes.

	The MDataBlock will provide smart handles for reading and writing this node's
	attribute values. Only these values should be used when performing computations.

	When evaluating the dependency graph, Maya will first call the compute method for
	this node. If the plug that is provided to the compute indicates that the attribute
	was defined by the Maya parent node, the compute method should return
	MS::kUnknownParameter. When this occurs, Maya will call the internal Maya node from
	which the user-defined node is derived to compute the plug's value.

	This means that a user defined node does not need to be concerned with computing
	inherited output attributes. However, if desired, these can be safely recomputed by
	this method to change the behaviour of the node.

	Args:
		plug (MPlug&): Plug representing the attribute that needs to be recomputed.
		dataBlock (MDataBlock&): Data block containing storage for the node's attributes.

	Returns:
		status code (MStatus): kSuccess if the operation was successful, kFailure if an	error occured
			during the operation.

	*/
	MStatus status;

	CHECK_MSTATUS_AND_RETURN_IT(parseDataBlock(dataBlock));
	CHECK_MSTATUS_AND_RETURN_IT(solveLimb());
	CHECK_MSTATUS_AND_RETURN_IT(updateOutput(plug, dataBlock));

	return MS::kSuccess;
}


MStatus Ik2bSolver::setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) {
	/* Sets the relation between attributes and marks the specified plugs dirty.

	Args:
		plugBeingDirtied (&MPlug): Plug which is being set dirty by Maya.
		affectedPlugs (&MPlugArray): The programmer should add any plugs which they want to set dirty
			to this list.

	*/
	// Rotation output
	if ( plugBeingDirtied == attrInFkIk
		|| plugBeingDirtied == attrInFkStart
		|| plugBeingDirtied == attrInFkMid
		|| plugBeingDirtied == attrInFkEnd
		|| plugBeingDirtied == attrInIkStart
		|| plugBeingDirtied == attrInIkMid
		|| plugBeingDirtied == attrInIkEnd
		|| plugBeingDirtied == attrInIkHandle
		|| plugBeingDirtied == attrInPv
		|| plugBeingDirtied == attrInTwist
		|| plugBeingDirtied == attrInSoftness
	)	{
		affectedPlugs.append(MPlug(objSelf, attrOutStart));
		affectedPlugs.append(MPlug(objSelf, attrOutMid));
		affectedPlugs.append(MPlug(objSelf, attrOutEnd));
		affectedPlugs.append(MPlug(objSelf, attrOutUpdate));
	}
	// Visibility output
	if (plugBeingDirtied == attrInFkIk)	{
		affectedPlugs.append(MPlug(objSelf, attrOutFkVisibility));
		affectedPlugs.append(MPlug(objSelf, attrOutIkVisibility));
	}

	return MS::kSuccess;
}


void Ik2bSolver::getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const {
	/* Disables Cached Playback support by default.

	Built-in locators all enable Cached Playback by default, but plug-ins have to
	explicitly enable it by overriding this method.
	This method should be overridden to enable Cached Playback by default for custom locators.

	Args:
		evalNode (MEvaluationNode&): This node's evaluation node, contains animated plug information
		disablingInfo (MNodeCacheDisablingInfo&): Information about why the node disables caching to be reported to the user
		cacheSetup (MNodeCacheSetupInfo&): Preferences and requirements this node has for caching
		monitoredAttribures (MObjectArray&): Attributes impacting the behavior of this method that will be monitored for change

	*/
	MPxNode::getCacheSetup(evalNode, disablingInfo, cacheSetupInfo, monitoredAttributes);
	assert(!disablingInfo.getCacheDisabled());
	cacheSetupInfo.setPreference(MNodeCacheSetupInfo::kWantToCacheByDefault, true);
}


void Ik2bSolver::postConstructor() {
	/* Post constructor.

	Internally maya creates two objects when a user defined node is created, the internal MObject and
	the user derived object. The association between the these two objects is not made until after the
	MPxNode constructor is called. This implies that no MPxNode member function can be called from the
	MPxNode constructor. The postConstructor will get called immediately after the constructor when it
	is safe to call any MPxNode member function.

	Reimplemented in MPxTransform, and MPxPolyTrg.

	*/
	objSelf = thisMObject();
}
