#pragma once

// System includes
#include <math.h>
#include <cassert>
#include <memory>
#include <unordered_map>
#include <atomic>

#include <maya/MUserData.h>
#include <maya/MMatrix.h>
#include <maya/MString.h>
#include <maya/MTypeId.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MColor.h>
#include <maya/MDistance.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MGlobal.h>
#include <maya/MDagMessage.h>
#include <maya/MEvaluationNode.h>
#include <maya/MEventMessage.h>
#include <maya/MEvaluationManager.h>
#include <maya/MEvaluationNode.h>
#include <maya/MDagModifier.h>
#include <maya/MSyntax.h>
#include <maya/MArgDatabase.h>

// Function sets
#include <maya/MFnDependencyNode.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>

// Viewport 2.0 includes
#include <maya/MDrawRegistry.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MShaderManager.h>
#include <maya/MHWGeometry.h>
#include <maya/MHWGeometryUtilities.h>

// Proxies
#include <maya/MPxTransform.h>
#include <maya/MPxTransformationMatrix.h>
#include <maya/MPxCommand.h>
#include <maya/MPxSurfaceShape.h>
#include <maya/MPxDrawOverride.h>

// Lunar
#include "../maya/api/Utils.h"
#include "../maya/api/MText.h"

#include "../maya/api/MAttribute.h"
#include "../maya/api/Utils.h"



class DegreeRadianConverter {
public:
	double degreesToRadians(double degrees) {return degrees * ( M_PI/ 180.0 );};
	double radiansToDegrees(double radians) {return radians * (180.0/M_PI);};
};


//-------------------------------------------------------------------------------------------------
//
// Component Transform Node definition
//-------------------------------------------------------------------------------------------------

class SpaceSwitchMatrix : public MPxTransformationMatrix {
public:
	// A really simple implementation of MPxTransformationMatrix.
	// - The virtual asMatrix() method which passes the matrix 
	// back to Maya when the command "xform -q -ws -m" is invoked
	static const MTypeId	type_id;

	double rockXValue;

	// Constructors
	SpaceSwitchMatrix()
		: MPxTransformationMatrix()
		, rockXValue(0.0)
	{};
	static MPxTransformationMatrix *creator() {return new SpaceSwitchMatrix();};
	
	MMatrix asMatrix() const override {return ParentClass::asMatrix();};
	MMatrix	asMatrix(double percent) const override;
	MMatrix	asRotateMatrix() const override;
	MStatus setRotatePivot(const MPoint&, MSpace::Space=MSpace::kTransform, bool balance=true) override;
	MQuaternion	preRotation() const override;

	// Degrees
	// double  getRockInX() const {return rockXValue;};
	// void    setRockInX(double rock) {rockXValue = rock;};

protected:	
	typedef MPxTransformationMatrix ParentClass;
	// Degree
};

// Class attributes
const MTypeId SpaceSwitchMatrix::type_id = 0x9000000;


MMatrix SpaceSwitchMatrix::asMatrix(double percent) const {

	MPxTransformationMatrix m(*this);

	//	Apply the percentage to the matrix components
	MVector trans = m.translation();
	trans *= percent;
	m.translateTo( trans );
	MPoint rotatePivotTrans = m.rotatePivot();
	rotatePivotTrans = rotatePivotTrans * percent;
	m.setRotatePivot(rotatePivotTrans);
	MPoint scalePivotTrans = m.scalePivotTranslation();
	scalePivotTrans = scalePivotTrans * percent;
	m.setScalePivotTranslation(scalePivotTrans);

	//	Apply the percentage to the rotate value.  Same
	// as above + the percentage gets applied
	MQuaternion quat = rotation();
	DegreeRadianConverter conv;
	double newTheta = conv.degreesToRadians(rockXValue);
	quat.setToXAxis(newTheta);
	m.rotateBy(quat);
	MEulerRotation eulRotate = m.eulerRotation();
	m.rotateTo(eulRotate * percent, MSpace::kTransform);

	//	Apply the percentage to the scale
	MVector s(scale(MSpace::kTransform));
	s.x = 1.0 + (s.x - 1.0)*percent;
	s.y = 1.0 + (s.y - 1.0)*percent;
	s.z = 1.0 + (s.z - 1.0)*percent;
	m.scaleTo(s, MSpace::kTransform);

	return m.asMatrix();
}


MMatrix	SpaceSwitchMatrix::asRotateMatrix() const {
	MMatrix Ro = rotateOrientationValue.asMatrix();
	MMatrix R  = rotationValue.asMatrix();
	MMatrix Rr = preRotation().asMatrix();

	MMatrix Rt;
	Rt[3][0] = rotatePivotTranslationValue.x;
	Rt[3][1] = rotatePivotTranslationValue.y;
	Rt[3][2] = rotatePivotTranslationValue.z;

	MMatrix Rp;
	Rp[3][0] = rotatePivotValue.x;
	Rp[3][1] = rotatePivotValue.y;
	Rp[3][2] = rotatePivotValue.z;

	MMatrix RpInv;
	RpInv[3][0] = -rotatePivotValue.x;
	RpInv[3][1] = -rotatePivotValue.y;
	RpInv[3][2] = -rotatePivotValue.z;

	return (RpInv * Ro * R * Rr * Rp * Rt);
}


MStatus SpaceSwitchMatrix::setRotatePivot(const MPoint &rotatePivot, MSpace::Space space, bool balance) {
	MPoint newPivot(rotatePivot);
	if (space != MSpace::kTransform) {
		if (space == MSpace::kPostTransform) {
			newPivot *= asMatrixInverse();
		}
		newPivot *= asScaleMatrix();
	}

	if (balance) {
		MMatrix Ro = rotateOrientationValue.asMatrix();
		MMatrix R = rotationValue.asMatrix();
		MMatrix Rr = preRotation().asMatrix();

		MMatrix Rp;
		Rp[3][0] = newPivot.x;
		Rp[3][1] = newPivot.y;
		Rp[3][2] = newPivot.z;

		MMatrix RpInv;
		RpInv[3][0] = -newPivot.x;
		RpInv[3][1] = -newPivot.y;
		RpInv[3][2] = -newPivot.z;

		MMatrix leftMat = RpInv * Ro * R * Rr * Rp;

		MMatrix mat = leftMat.inverse() * asRotateMatrix();

		rotatePivotTranslationValue[0] = mat[3][0];
		rotatePivotTranslationValue[1] = mat[3][1];
		rotatePivotTranslationValue[2] = mat[3][2];
	}

	rotatePivotValue = newPivot;
	return MS::kSuccess;
}

// This method returns the local rotation used by rotate manipulator
MQuaternion SpaceSwitchMatrix::preRotation() const {
	DegreeRadianConverter conv;
	double newTheta = conv.degreesToRadians(rockXValue);
	MQuaternion quat; quat.setToXAxis(newTheta);
	return quat;
}






class ComponentNode : public MPxTransform {
	/* Component Node - transform instance with a custom type_name. */
public:
	// Class attributes
	static const MString type_name;
	static const MTypeId type_id;
	static const MString type_drawdb;
	static const MString type_drawid;

	// Space switch attributes
	// static MObject attr_enable_spaces;
	// static MObject attr_space_indx;
	// static MObject attr_offset_mat;
	// static MObject attr_driver_mat;
	// static MObject attr_driverinv_mat;
	static MObject attr_ctrls;
	static MObject attr_fk;
	static MObject attr_ik;

	MObject self_object;

	// Constructors
	ComponentNode()
		: MPxTransform()
	{};
	// Destructors
	virtual ~ComponentNode() override {};

	// Class Methods
	static void* 		creator() {return new ComponentNode();};
	static MStatus	initialize();
	virtual void 		postConstructor() override;
	// virtual MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;

	void 					  getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const override;
	SchedulingType  schedulingType() const override {return SchedulingType::kParallel;}

	// void  	resetTransformation (MPxTransformationMatrix* matrix) override {MPxTransform::resetTransformation(matrix);};
	// void  	resetTransformation (const MMatrix& resetMatrix) override {MPxTransform::resetTransformation(resetMatrix);};
	// MStatus validateAndSetValue(const MPlug&, const MDataHandle&) override;

};


// Class attributes
const MString ComponentNode::type_name 		= "component";
const MTypeId ComponentNode::type_id 			= 0x9000001;
const MString ComponentNode::type_drawdb	= "drawdb/geometry/animation/component";
const MString ComponentNode::type_drawid	= "componentPlugin";

MObject ComponentNode::attr_ctrls;
MObject ComponentNode::attr_fk;
MObject ComponentNode::attr_ik;

// MObject ComponentNode::attr_enable_spaces;
// MObject ComponentNode::attr_space_indx;
// MObject ComponentNode::attr_offset_mat;
// MObject ComponentNode::attr_driver_mat;
// MObject ComponentNode::attr_driverinv_mat;
// MObject ComponentNode::attr_spaces;


// MStatus ComponentNode::initialize() {
// 	MFnUnitAttribute 		 fn_unit;
// 	MFnNumericAttribute  fn_num;
// 	MFnEnumAttribute 		 fn_enum;
// 	MFnCompoundAttribute fn_comp;
// 	MFnMatrixAttribute 	 fn_mat;
// 	MStatus status;

// 	attr_enable_spaces = fn_num.create("enableSpaces", "enas", MFnNumericData::kBoolean, false);
// 	fn_num.setStorable(true);
// 	fn_num.setKeyable(true);
// 	fn_num.setChannelBox(true);
// 	fn_num.setAffectsWorldSpace(true);

// 	attr_space_indx = fn_enum.create("space", "spac");
// 	fn_enum.addField("World", 0);
// 	fn_enum.addField("Main", 1);
// 	fn_enum.addField("Root", 2);
// 	fn_enum.addField("Pelvis", 3);
// 	fn_enum.addField("Component", 4);
// 	fn_enum.setKeyable(true);
// 	fn_enum.setAffectsWorldSpace(true);

// 	/* spacesAttr:
// 	-- enableSpaces
// 	-- space
// 	-- spaces
// 		| -- offsetMatrix
// 		| -- driverMatrix
// 		| -- driverInverseMatrix
// 	*/
// 	attr_offset_mat = fn_mat.create("offsetMatrix", "ofm");
// 	fn_mat.setKeyable(true);
// 	fn_mat.setReadable(false);
// 	fn_mat.setAffectsWorldSpace(true);

// 	attr_driver_mat = fn_mat.create("driverMatrix", "drm");
// 	fn_mat.setKeyable(true);
// 	fn_mat.setReadable(false);
// 	fn_mat.setAffectsWorldSpace(true);

// 	attr_driverinv_mat = fn_mat.create("driverInverseMatrix", "dim");
// 	fn_mat.setKeyable(true);
// 	fn_mat.setReadable(false);
// 	fn_mat.setAffectsWorldSpace(true);

// 	attr_spaces = fn_comp.create("spaces", "spcs");
// 	fn_comp.addChild(attr_offset_mat);
// 	fn_comp.addChild(attr_driver_mat);
// 	fn_comp.addChild(attr_driverinv_mat);
// 	fn_comp.setArray(true);
// 	fn_comp.setKeyable(true);
// 	fn_comp.setReadable(false);
// 	fn_comp.setAffectsWorldSpace(true);

// 	// Add attributes
// 	addAttributes(attr_enable_spaces,	attr_space_indx, attr_spaces);

// 	attributeAffects(attr_enable_spaces, matrix);
// 	attributeAffects(attr_space_indx, matrix);
// 	attributeAffects(attr_offset_mat, matrix);
// 	attributeAffects(attr_driver_mat, matrix);
// 	attributeAffects(attr_driverinv_mat, matrix);
// 	attributeAffects(attr_spaces, matrix);

// 	// This is required so that the validateAndSet method is called
// 	mustCallValidateAndSet(attr_enable_spaces);
// 	mustCallValidateAndSet(attr_space_indx);
// 	mustCallValidateAndSet(attr_offset_mat);
// 	mustCallValidateAndSet(attr_driver_mat);
// 	mustCallValidateAndSet(attr_driverinv_mat);
// 	mustCallValidateAndSet(attr_spaces);

// 	return MS::kSuccess;
// }

MStatus ComponentNode::initialize() {
	/* Node Initializer.

	This method initializes the node, and should be overridden in user-defined nodes.
	
	Returns:
		status code (MStatus): kSuccess if the operation was successful, kFailure if an	error occured
			during the operation.

	*/
	MFnUnitAttribute fn_unit;
	MFnNumericAttribute fn_num;
	MFnEnumAttribute fn_enum;
	MFnMessageAttribute fn_mess;
	MFnCompoundAttribute fn_comp;
	MStatus status;

	attr_fk = fn_mess.create("fk", "fk");
	fn_mess.setWritable(true);
	fn_mess.setReadable(false);
	fn_mess.setArray(true);

	attr_ik = fn_mess.create("ik", "ik");
	fn_mess.setWritable(true);
	fn_mess.setReadable(false);
	fn_mess.setArray(true);

	attr_ctrls = fn_comp.create("ctrls", "ctrls");
	fn_comp.addChild(attr_fk);
	fn_comp.addChild(attr_ik);
	// fn_comp.setArray(false);
	fn_mess.setWritable(true);
	fn_comp.setReadable(false);
	// fn_comp.setAffectsWorldSpace(true);

	addAttributes(attr_ctrls);

}

void ComponentNode::postConstructor() {
	self_object=thisMObject();
	MFnDependencyNode fn_this(self_object);
	fn_this.findPlug("shear", false).setLocked(true);
}


void ComponentNode::getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const {
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
	MPxTransform::getCacheSetup(evalNode, disablingInfo, cacheSetupInfo, monitoredAttributes);
	assert(!disablingInfo.getCacheDisabled());
	cacheSetupInfo.setPreference(MNodeCacheSetupInfo::kWantToCacheByDefault, true);
}


// MStatus ComponentNode::validateAndSetValue(const MPlug& plug, const MDataHandle& handle) {
	
// 	if (plug.isNull()) {return MS::kFailure;}

// 	if ( plug == attr_enable_spaces
// 		|| plug == attr_space_indx
// 		|| plug == attr_offset_mat
// 		|| plug == attr_driver_mat
// 		|| plug == attr_driverinv_mat
// 		|| plug == attr_spaces
// 	) {
// 		MStatus status;
// 		MDataBlock dataBlock = forceCache();
// 		// We need to grab some data from the input plug and handle
// 		MDataHandle blockHandle = dataBlock.outputValue(plug, &status);
// 		blockHandle.set(handle.asDouble());
// 		blockHandle.setClean();

// 		// MDataHandle dh_enable_spaces = dataBlock.outputValue(attr_enable_spaces, &status);
// 		// bool enable_spaces = dh_enable_spaces.asBool();
// 		// // dh_enable_spaces.setBool(enable_spaces);
// 		// if (enable_spaces) {
// 		MDataHandle dh_space_indx = dataBlock.outputValue(attr_space_indx, &status);
// 		short space_indx = dh_space_indx.asShort();
// 		MArrayDataHandle dh_spaces = dataBlock.outputArrayValue(attr_spaces, &status);
// 		dh_spaces.jumpToArrayElement(space_indx);
// 		MDataHandle dh_space_at_indx = dh_spaces.outputValue();
// 		MDataHandle dh_offset_mat = dh_space_at_indx.child(attr_offset_mat);
// 		MDataHandle dh_driver_mat = dh_space_at_indx.child(attr_driver_mat);
// 		MDataHandle dh_driverinv_mat = dh_space_at_indx.child(attr_driverinv_mat);
// 		MMatrix mat_offset = dh_offset_mat.asMatrix();
// 		MMatrix mat_driver = dh_driver_mat.asMatrix();
// 		MMatrix mat_driverinv = dh_driverinv_mat.asMatrix();

// 		// dh_enable_spaces.setClean();
// 		// dh_space_indx.setShort(space_indx);
// 		// dh_space_indx.setClean();
// 		// dh_offset_mat.setMMatrix(mat_offset);
// 		// dh_offset_mat.setClean();
// 		// dh_driver_mat.setMMatrix(mat_driver);
// 		// dh_driver_mat.setClean();
// 		// dh_driverinv_mat.setMMatrix(mat_driverinv);
// 		// dh_driverinv_mat.setClean();

// 		MDataHandle dh_offset_parent_mat = dataBlock.outputValue(offsetParentMatrix, &status);
// 		dh_offset_parent_mat.setMMatrix(mat_offset * mat_driver * mat_driverinv);
// 		dh_offset_parent_mat.setClean();
// 		// dh_spaces.setAllClean();

// 		// Mark the matrix as dirty so that DG information will update.
// 		dirtyMatrix();
// 		// }
// 		return status;
// 	}

// 	// Allow processing for other attributes
// 	return MPxTransform::validateAndSetValue(plug, handle);
// }


// MStatus ComponentNode::compute(const MPlug& plug, MDataBlock& dataBlock) {
// 	/* */
// 	if ( plug == matrix
// 		|| plug == inverseMatrix
// 		|| plug == worldMatrix
// 		|| plug == worldInverseMatrix
// 		|| plug == parentMatrix
// 		|| plug == parentInverseMatrix
// 	)	{
// 		MDataHandle dh_enable_spaces = dataBlock.inputValue(attr_enable_spaces);
// 		bool enable_spaces = dh_enable_spaces.asBool();
// 		if (enable_spaces) {
// 			MDataHandle dh_space_indx = dataBlock.inputValue(attr_space_indx);
// 			short space_indx = dh_space_indx.asShort();
// 			MArrayDataHandle dh_spaces = dataBlock.inputArrayValue(attr_spaces);
// 			dh_spaces.jumpToArrayElement(space_indx);
// 			MDataHandle dh_space_at_indx = dh_spaces.inputValue();

// 			MMatrix mat_offset = dh_space_at_indx.child(attr_offset_mat).asMatrix();
// 			MMatrix mat_driver = dh_space_at_indx.child(attr_driver_mat).asMatrix();
// 			MMatrix mat_driverinv = dh_space_at_indx.child(attr_driverinv_mat).asMatrix();

// 			MDataHandle dh_offset_parent_mat = dataBlock.outputValue(offsetParentMatrix);
// 			dh_offset_parent_mat.setMMatrix(mat_offset * mat_driver * mat_driverinv);
// 			dh_offset_parent_mat.setClean();

// 			return MS::kSuccess;
// 		}
// 	}
// 	return MPxTransform::compute(plug, dataBlock);
// }






//-------------------------------------------------------------------------------------------------
//
// Component Transform Node definition
//
//-------------------------------------------------------------------------------------------------


class ComponentCmd : public MPxCommand {

public:

	enum CommandMode {kCommandCreate, kCommandHelp};
	CommandMode command;

	// Public Data
	static const char* command_name;

	// Command's Flags
	static const char* fs_name;
	static const char* fl_name;
	static const char* fs_parent;
	static const char* fl_parent;
	static const char* fs_lock_attributes;
	static const char* fl_lock_attributes;
	static const char* fs_help;
	static const char* fl_help;

	MString name;
	MString parent;
	bool lock_attributes;

	// Constructors
	ComponentCmd()
 		: MPxCommand()
		, name(ComponentNode::type_name)
		, lock_attributes(false)
		, command(kCommandCreate)
	{};

	// Inherited Public Methods
	static void* creator() {return new ComponentCmd();}
	virtual bool isUndoable() const override {return command == kCommandCreate;}
	static MSyntax syntaxCreator();

	virtual MStatus doIt(const MArgList& argList) override;
	virtual MStatus redoIt() override;
	virtual MStatus undoIt() override;

	MStatus parseArguments(const MArgList& argList);

private:
	MObject self_object;
	MDagPath self_dp;

	MSelectionList list_sel;
	MDagModifier mod_dag;
};


// Public Data
const char* ComponentCmd::command_name = "component";

// Command's Flags
const char* ComponentCmd::fs_name = "-n";
const char* ComponentCmd::fl_name = "-name";
const char* ComponentCmd::fs_parent = "-p";
const char* ComponentCmd::fl_parent = "-parent";
const char* ComponentCmd::fs_lock_attributes = "-la";
const char* ComponentCmd::fl_lock_attributes = "-lockAttributes";
const char* ComponentCmd::fs_help = "-h";
const char* ComponentCmd::fl_help = "-help";


MSyntax ComponentCmd::syntaxCreator() {
	/* Creates the command's syntax object and returns it.

	Returns:
		syntax (MSyntax): Command's syntax object

	*/
	MSyntax sytnax;

	// Main flags
	sytnax.addFlag(fs_name, fl_name, MSyntax::kString);
	sytnax.addFlag(fs_parent, fl_parent, MSyntax::kString);
	sytnax.addFlag(fs_lock_attributes, fl_lock_attributes, MSyntax::kBoolean);
	sytnax.addFlag(fs_help, fl_help, MSyntax::kBoolean);

	sytnax.setObjectType(MSyntax::kSelectionList, 0, 1);
	sytnax.useSelectionAsDefault(true);

	return sytnax;
}


MStatus ComponentCmd::parseArguments(const MArgList &argList) {
	/* Parses the commands's flag arguments.

	Args:
		argList (MArglist): List of arguments passed to the command.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	MArgDatabase argData(syntax(), argList);
	argData.getObjects(list_sel);

	// Display Help
	if (argData.isFlagSet(fs_help))	{
		command = kCommandHelp;
		MString strHelp;
		strHelp += "Flags:\n";
		strHelp += "   -n     -name                 String     Name of the rig controller to create.\n";
		strHelp += "   -p     -parent               String     Name of the object that will be the parent.\n";
		strHelp += "   -la    -lockAttributes       String     Whether or not to lock the transform attributes.\n";
		strHelp += "   -h     -help                 N/A        Display this text.\n";
		MGlobal::displayInfo(strHelp);
		return MS::kSuccess;
	}
	// Name Flag
	if (argData.isFlagSet(fs_name))	{
		name = argData.flagArgumentString(fs_name, 0, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}
	// Parent Flag
	if (argData.isFlagSet(fs_parent)) {
		list_sel.add(argData.flagArgumentString(fs_parent, 0, &status));
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}
	// Lock shape attributes
	if (argData.isFlagSet(fs_lock_attributes)) {
		lock_attributes = argData.flagArgumentBool(fs_lock_attributes, 0, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}

	return MS::kSuccess;
}


MStatus ComponentCmd::doIt(const MArgList& argList) {
	/* Command's doIt method.

	This method should perform a command by setting up internal class data and then	calling the
	redoIt method.

	The actual action performed by the command should be done in the redoIt method.	This is a pure
	virtual method, and must be overridden in derived classes.

	Args:
		argList (MArgList): List of arguments passed to the command.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	status = parseArguments(argList);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	// Command create mode
	if (command == kCommandCreate) {
		self_object = mod_dag.createNode(ComponentNode::type_name, MObject::kNullObj);
		if (name != ComponentNode::type_name) {mod_dag.renameNode(self_object, name);}
		// Parent under the transform node if the selection is not empty and / or parent was specified
		int numItems = list_sel.length();
		if (numItems == 1) {
			MObject parent_object;
			list_sel.getDependNode(0, parent_object);
			mod_dag.reparentNode(self_object, parent_object);
		}	else if (numItems == 2)	{
			MObject parent_object;
			list_sel.getDependNode(1, parent_object);
			mod_dag.reparentNode(self_object, parent_object);
		}
	}

	return redoIt();
}


MStatus ComponentCmd::redoIt() {
	/* Command's redoIt method.

	This method should do the actual work of the command based on the internal class data only.	Internal class data should be set in the doIt method.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	// Command create mode
	if (command == kCommandCreate) {
		MStatus status;
		// We need to init the MFnTransform with a dag path, mobjects do not work with transformations
		// even if the object has a MFn::kTransform
		MDagPath::getAPathTo(self_object, self_dp);
		MFnTransform fn_transform(self_dp);

		status = mod_dag.doIt();
		CHECK_MSTATUS_AND_RETURN_IT(status);

		// TRANSFORM NODE
		// if (bTranslateTo == true) {fn_transform.setTranslation(posTarget, MSpace::kWorld);}
		// if (bRotateTo == true) {fn_transform.setRotation(rotTarget, MSpace::kWorld);}
	
		// MPlug plug_has_dynamic_attributes = fn_transform.findPlug("hasDynamicAttributes", false);
		// plug_has_dynamic_attributes.setValue(has_dynamic_attributes);

		// Lock shape attributes
		if (lock_attributes == true) {	
			MAttribute::lockAndHideAttr(self_object, ComponentNode::translate);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::rotate);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::scale);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::shear);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::rotateAxis);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::rotateOrder);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::inheritsTransform);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::offsetParentMatrix);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::rotateQuaternion);
			MAttribute::lockAndHideAttr(self_object, ComponentNode::visibility);
		}

		// Set hide on playback
		// fn_transform.findPlug("hideOnPlayback", false).setValue(bHideOnPlayback);

		// Sets command's output result in mel / python
		clearResult();
		setResult(fn_transform.name());
		// appendToResult(fn_transform.name());
	}

	return MS::kSuccess;
}


MStatus ComponentCmd::undoIt() {
	/* Command's undoIt method.

	This method should undo the work done by the redoIt method based on the internal class data only.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	// Restore the initial state
	status = mod_dag.undoIt();
	CHECK_MSTATUS_AND_RETURN_IT(status);

	return MS::kSuccess;
}

