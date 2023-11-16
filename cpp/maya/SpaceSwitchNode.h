#pragma once

// System includes
#include <string.h>
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
#include <maya/MArrayDataHandle.h>

// Function sets
#include <maya/MFnDependencyNode.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnCompoundAttribute.h>
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






class SpaceSwitchNode : public MPxNode {
/* Component Node - transform instance with a custom type_name. */
public:
	// Class attributes
	static const MString type_name;
	static const MTypeId type_id;
	static const MString type_drawdb;
	static const MString type_drawid;

	// Space switch attributes
	static MObject attr_space_indx;
	static MObject attr_offset_mat;
	static MObject attr_driver_mat;
	static MObject attr_driverinv_mat;
	static MObject attr_spaces;

	static MObject attr_out_mat;

	// Node plugs for dirty marking - optimization
	MPlug plug_out_mat;
	MObject obj_spaceswitch;

	// Constructors
	SpaceSwitchNode()
		: MPxNode()
	{};
	// Destructors
	virtual ~SpaceSwitchNode() override {};

	// Class Methods
	static void * 	creator() {return new SpaceSwitchNode();};
	static MStatus	initialize();
	virtual void 		postConstructor() override;
	virtual MStatus compute(const MPlug& plug, MDataBlock& dataBlock) override;
	MStatus setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) override;

	void 					  getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const override;
	SchedulingType  schedulingType() const override {return SchedulingType::kParallel;}

};


// Class attributes
const MString SpaceSwitchNode::type_name 	= "spaceSwitch";
const MTypeId SpaceSwitchNode::type_id 		= 0x9000030;

// MObject SpaceSwitchNode::attr_enable_spaces;
MObject SpaceSwitchNode::attr_space_indx;
MObject SpaceSwitchNode::attr_offset_mat;
MObject SpaceSwitchNode::attr_driver_mat;
MObject SpaceSwitchNode::attr_driverinv_mat;
MObject SpaceSwitchNode::attr_spaces;

MObject SpaceSwitchNode::attr_out_mat;


MStatus SpaceSwitchNode::initialize() {
	/* Node Initializer.

	This method initializes the node, and should be overridden in user-defined nodes.

	|-- spaces
	|   |-- offsetMatrix
	|   |-- driverMatrix
	|   |-- driverInverseMatrix
	|-- etc
	|   |-- file3
	|   `-- directory1
	|       |-- file4
	|       `-- file5


	Returns:
		status code (MStatus): kSuccess if the operation was successful, kFailure if an	error occured
			during the operation.

	*/
	MFnUnitAttribute 		 fn_unit;
	MFnNumericAttribute  fn_num;
	MFnEnumAttribute 		 fn_enum;
	MFnCompoundAttribute fn_comp;
	MFnMatrixAttribute 	 fn_mat;
	MStatus status;

	attr_space_indx = fn_enum.create("space", "spac");
	fn_enum.addField("World", 0);
	fn_enum.addField("Main", 1);
	fn_enum.addField("Root", 2);
	fn_enum.addField("Pelvis", 3);
	fn_enum.addField("Component", 4);
	fn_enum.setKeyable(true);
	fn_enum.setAffectsWorldSpace(true);

	/* spacesAttr:
	-- enableSpaces
	-- space
	-- spaces
		| -- offsetMatrix
		| -- driverMatrix
		| -- driverInverseMatrix
	*/
	attr_offset_mat = fn_mat.create("offsetMatrix", "ofm");
	fn_mat.setKeyable(true);
	fn_mat.setReadable(false);
	fn_mat.setAffectsWorldSpace(true);

	attr_driver_mat = fn_mat.create("driverMatrix", "drm");
	fn_mat.setKeyable(true);
	fn_mat.setReadable(false);
	fn_mat.setAffectsWorldSpace(true);

	attr_driverinv_mat = fn_mat.create("driverInverseMatrix", "dim");
	fn_mat.setKeyable(true);
	fn_mat.setReadable(false);
	fn_mat.setAffectsWorldSpace(true);

	attr_spaces = fn_comp.create("spaces", "spcs");
	fn_comp.addChild(attr_offset_mat);
	fn_comp.addChild(attr_driver_mat);
	fn_comp.addChild(attr_driverinv_mat);
	fn_comp.setArray(true);
	fn_comp.setKeyable(true);
	fn_comp.setReadable(false);
	fn_comp.setAffectsWorldSpace(true);

	attr_out_mat = fn_mat.create("outputMatrix", "outm");
	fn_mat.setKeyable(true);
	fn_mat.setWritable(false);
	fn_mat.setReadable(true);

	// Add attributes
	addAttributes(attr_space_indx, attr_spaces, attr_out_mat);

	attributeAffects(attr_space_indx, attr_out_mat);
	attributeAffects(attr_offset_mat, attr_out_mat);
	attributeAffects(attr_driver_mat, attr_out_mat);
	attributeAffects(attr_driverinv_mat, attr_out_mat);
	attributeAffects(attr_spaces, attr_out_mat);

	return MS::kSuccess;
}


void SpaceSwitchNode::postConstructor() {
	obj_spaceswitch=thisMObject();

	plug_out_mat = MPlug(obj_spaceswitch, attr_out_mat);
}


void SpaceSwitchNode::getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const {
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



MStatus SpaceSwitchNode::setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) {
	/* Sets the relation between attributes and marks the specified plugs dirty.

	setDependentsDirty -> compute
	setDependentsDirty -> validateAndSetValue

	Args:
		plugBeingDirtied (&MPlug): Plug which is being set dirty by Maya.
		affectedPlugs (&MPlugArray): The programmer should add any plugs which they want to set dirty
			to this list.

	*/
	// Rotation output
	if ( plugBeingDirtied == attr_space_indx
		|| plugBeingDirtied == attr_offset_mat
		|| plugBeingDirtied == attr_driver_mat
		|| plugBeingDirtied == attr_driverinv_mat
		|| plugBeingDirtied == attr_spaces
	)	{
		affectedPlugs.append(plug_out_mat);
	}
	return MS::kSuccess;
}



MStatus SpaceSwitchNode::compute(const MPlug& plug, MDataBlock& dataBlock) {

	if (plug != attr_out_mat) {return MS::kUnknownParameter;}
	MStatus status;

	MDataHandle dh_space_indx = dataBlock.inputValue(attr_space_indx, &status);
	short space_indx = dh_space_indx.asShort();
	MArrayDataHandle dh_spaces = dataBlock.inputArrayValue(attr_spaces, &status);
	dh_spaces.jumpToArrayElement(space_indx);
	MDataHandle dh_space_at_indx = dh_spaces.inputValue();

	MMatrix mat_offset = dh_space_at_indx.child(attr_offset_mat).asMatrix();
	MMatrix mat_driver = dh_space_at_indx.child(attr_driver_mat).asMatrix();
	MMatrix mat_driverinv = dh_space_at_indx.child(attr_driverinv_mat).asMatrix();

	MDataHandle dh_out_matrix = dataBlock.outputValue(attr_out_mat, &status);
	dh_out_matrix.setMMatrix(mat_offset * mat_driver * mat_driverinv);
	dh_out_matrix.setClean();

	dataBlock.setClean(plug);

	return MS::kSuccess;
}






//-------------------------------------------------------------------------------------------------
//
// Component Transform Node definition
//
//-------------------------------------------------------------------------------------------------


class SpaceSwitchCmd : public MPxCommand {
public:
	enum CommandMode {kCommandCreate, kCommandHelp};
	CommandMode command;

	// Public Data
	static const char* command_name;

	// Command's Flags
	static const char* fs_name;
	static const char* fl_name;
	static const char* fs_object;
	static const char* fl_object;
	static const char* fs_help;
	static const char* fl_help;

	MString name;
	MString object;
	bool lock_attributes;

	// Constructors
	SpaceSwitchCmd()
 		: MPxCommand()
		, name(SpaceSwitchNode::type_name)
		, lock_attributes(false)
		, command(kCommandCreate)
	{};

	// Inherited Public Methods
	static void* creator() {return new SpaceSwitchCmd();}
	virtual bool isUndoable() const override {return command == kCommandCreate;}
	static MSyntax syntaxCreator();

	virtual MStatus doIt(const MArgList& argList) override;
	virtual MStatus redoIt() override;
	virtual MStatus undoIt() override;

	MStatus parseArguments(const MArgList& argList);

	MObject obj_spaceswitch;
	MObject obj_driven;

	MDagPath self_dp;
	MSelectionList list_sel;
	MDagModifier mod_dag;
	MDGModifier mod_dg;
};


// Public Data

const char* SpaceSwitchCmd::command_name = "spaceSwitch";

// Command's Flags
const char* SpaceSwitchCmd::fs_name = "-n";
const char* SpaceSwitchCmd::fl_name = "-name";
const char* SpaceSwitchCmd::fs_object = "-o";
const char* SpaceSwitchCmd::fl_object = "-object";
const char* SpaceSwitchCmd::fs_help = "-h";
const char* SpaceSwitchCmd::fl_help = "-help";


MSyntax SpaceSwitchCmd::syntaxCreator() {
	/* Creates the command's syntax object and returns it.

	Returns:
		syntax (MSyntax): Command's syntax object

	*/
	MSyntax sytnax;

	// Main flags
	sytnax.addFlag(fs_name, fl_name, MSyntax::kString);
	sytnax.addFlag(fs_object, fl_object, MSyntax::kString);
	sytnax.addFlag(fs_help, fl_help, MSyntax::kBoolean);

	sytnax.setObjectType(MSyntax::kSelectionList, 0, 100);
	sytnax.useSelectionAsDefault(true);

	return sytnax;
}


MStatus SpaceSwitchCmd::parseArguments(const MArgList &argList) {
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
		strHelp += "   -o     -object               String     Objects that will be used as space switches.\n";
		strHelp += "   -h     -help                 N/A        Display this text.\n";
		MGlobal::displayInfo(strHelp);
		return MS::kSuccess;
	}
	// Name Flag
	if (argData.isFlagSet(fs_name))	{
		name = argData.flagArgumentString(fs_name, 0, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}
	// Driven Flag
	if (argData.isFlagSet(fs_object)) {
		list_sel.add(argData.flagArgumentString(fl_object, 0, &status));
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}
	// Lock shape attributes
	// if (argData.isFlagSet(fs_lock_attributes)) {
	// 	lock_attributes = argData.flagArgumentBool(fs_lock_attributes, 0, &status);
	// 	CHECK_MSTATUS_AND_RETURN_IT(status);
	// }

	return MS::kSuccess;
}


MStatus SpaceSwitchCmd::doIt(const MArgList& argList) {
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
		obj_spaceswitch = mod_dg.createNode(SpaceSwitchNode::type_name);
		if (name != SpaceSwitchNode::type_name) {mod_dg.renameNode(obj_spaceswitch, name);}

		list_sel.getDependNode(-1, obj_driven);

		// Parent under the transform node if the selection is not empty and / or parent was specified
		int numItems = list_sel.length();
		// if (numItems == 1) {
		// 	// MObject parent_object;
		// 	// list_sel.getDependNode(0, parent_object);
		// 	// mod_dg.reparentNode(obj_spaceswitch, parent_object);
		// }	else if (numItems == 2)	{
		// 	// MObject parent_object;
		// 	// list_sel.getDependNode(1, parent_object);
		// 	// mod_dg.reparentNode(obj_spaceswitch, parent_object);
		// }
	}

	return redoIt();
}


MStatus SpaceSwitchCmd::redoIt() {
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
		MFnDependencyNode fn_spaceSwitch(obj_spaceswitch);
		MFnDependencyNode fn_driven(obj_driven);
		
		// MDagPath::getAPathTo(obj_spaceswitch, self_dp);
		// MFnTransform fn_transform(self_dp);

		status = mod_dg.doIt();
		CHECK_MSTATUS_AND_RETURN_IT(status);

		// TRANSFORM NODE
		// if (bTranslateTo == true) {fn_transform.setTranslation(posTarget, MSpace::kWorld);}
		// if (bRotateTo == true) {fn_transform.setRotation(rotTarget, MSpace::kWorld);}
	
		MPlug plug_out_mat = fn_spaceSwitch.findPlug("outputMatrix", false);
		MPlug plug_opm = fn_driven.findPlug("offsetParentMatrix", false);

		mod_dg.connect(plug_out_mat, plug_opm);
		mod_dg.doIt();

		// Lock shape attributes
		// if (lock_attributes == true) {
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::translate);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::rotate);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::scale);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::shear);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::rotateAxis);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::rotateOrder);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::inheritsTransform);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::offsetParentMatrix);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::rotateQuaternion);
			// LMAttribute::lockAndHideAttr(obj_spaceswitch, SpaceSwitchNode::visibility);
		// }

		// Set hide on playback
		// fn_transform.findPlug("hideOnPlayback", false).setValue(bHideOnPlayback);

		// Sets command's output result in mel / python
		clearResult();
		setResult(fn_spaceSwitch.name());
	}

	return MS::kSuccess;
}


MStatus SpaceSwitchCmd::undoIt() {
	/* Command's undoIt method.

	This method should undo the work done by the redoIt method based on the internal class data only.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	// Restore the initial state
	status = mod_dg.undoIt();
	CHECK_MSTATUS_AND_RETURN_IT(status);

	return MS::kSuccess;
}

