#include "CtrlNode.h"
#include "CtrlCmd.h"
#include "Ik2bSolver.h"
#include "IkCommand.h"
// #include "FootRollSolver.h"
// #include "MetaDataNode.h"
// #include "MetaDataCmd.h"
// #include "TwistSolver.h"
// #include "TwistSolver.h"

#include "ComponentNode.h"
#include "SpaceSwitchNode.h"
// #include "DenoiseNode.h"

// Function Sets
#include <maya/MFnPlugin.h>




// Callback variables
// static MCallbackIdArray callbackIds;
// static MCallbackId afterNewCallbackId;
// static MCallbackId afterOpenCallbackId;
// static MCallbackId afterSaveSetMetaDataNodeCbId;



void setMelConfig(void*) {
	/* Sets the selection priority for locators to 999. */
	MGlobal::executeCommandOnIdle("cycleCheck -e 0");
	// MGlobal::executeCommandOnIdle("selectPriority -transform 666");
}

// static void onSceneSaved(void* clientData) {
// 	MGlobal::executePythonCommand("import cgp.maya.api.Maya as nom");
// 	// MGlobal::executePythonCommand("lm.LMMetaData().setFromSceneName()");
// }




MStatus initializePlugin(MObject obj) {
	// Plugin variables
	const char* author = "Lunatics";
	const char* version = "0.1.0";
	const char* requiredApiVersion = "Any";

	MStatus status;
	MFnPlugin fn_plugin(obj, author, version, requiredApiVersion);

	// Register Controller node
	status = fn_plugin.registerTransform(
		ComponentNode::type_name,
		ComponentNode::type_id, 
		&ComponentNode::creator, 
		&ComponentNode::initialize,
		&MPxTransformationMatrix::creator,
		MPxTransformationMatrix::baseTransformationMatrixId,
		&ComponentNode::type_drawdb
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	status = fn_plugin.registerCommand(
		ComponentCmd::command_name,
		ComponentCmd::creator,
		ComponentCmd::syntaxCreator
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	status = fn_plugin.registerTransform(
		CtrlNode::type_name,
		CtrlNode::type_id, 
		&CtrlNode::creator, 
		&CtrlNode::initialize,
		&MPxTransformationMatrix::creator,
		MPxTransformationMatrix::baseTransformationMatrixId,
		&CtrlNode::type_drawdb
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
		CtrlNode::type_drawdb,
		CtrlNode::type_drawid,
		CtrlDrawOverride::creator
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	status = fn_plugin.registerCommand(
		CtrlCommand::commandName,
		CtrlCommand::creator,
		CtrlCommand::syntaxCreator
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	// Register Ik2bSolver node
	status = fn_plugin.registerNode(
		Ik2bSolver::typeName,
		Ik2bSolver::typeId,
		Ik2bSolver::creator,
		Ik2bSolver::initialize,
		MPxNode::kDependNode
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);
	// Register ik command
	status = fn_plugin.registerCommand(
		IkCommand::commandName,
		IkCommand::creator,
		IkCommand::syntaxCreator
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Register MetaData node
	// status = fn_plugin.registerNode(
	// 	MetaDataNode::type_name,
	// 	MetaDataNode::type_id,
	// 	MetaDataNode::creator,
	// 	MetaDataNode::initialize,
	//  	MPxLocatorNode::kLocatorNode,
	//  	&MetaDataNode::type_drawdb
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);
	// // Register MetaData draw override
	// status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
	// 	MetaDataNode::type_drawdb,
	// 	MetaDataNode::type_drawid,
	// 	MetaDataDrawOverride::creator
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);
	// // Register MetaData command
	// status = fn_plugin.registerCommand(
	// 	MetaDataCmd::commandName,
	// 	MetaDataCmd::creator,
	// 	MetaDataCmd::syntaxCreator
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Register FootRoll node
	// status = fn_plugin.registerNode(
	// 	FootRollSolver::typeName,
	// 	FootRollSolver::typeId,
	// 	FootRollSolver::creator,
	// 	FootRollSolver::initialize,
	// 	MPxNode::kDependNode
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Register TwistSolver node
	// status = fn_plugin.registerNode(
	// 	TwistSolver::typeName,
	// 	TwistSolver::typeId,
	// 	TwistSolver::creator,
	// 	TwistSolver::initialize,
	// 	MPxNode::kDependNode
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// Register space switch node
	status = fn_plugin.registerNode(
		SpaceSwitchNode::type_name,
		SpaceSwitchNode::type_id,
		SpaceSwitchNode::creator,
		SpaceSwitchNode::initialize,
		MPxNode::kDependNode
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	// Register Space switch command
	status = fn_plugin.registerCommand(
		SpaceSwitchCmd::command_name,
		SpaceSwitchCmd::creator,
		SpaceSwitchCmd::syntaxCreator
	);
	CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Register Denoise Node
	// status = fn_plugin.registerNode(
	// 	DenoiseNode::type_name,
	// 	DenoiseNode::type_id,
	// 	DenoiseNode::creator,
	// 	DenoiseNode::initialize,
	// 	MPxNode::kDependNode
	// );
	// CHECK_MSTATUS_AND_RETURN_IT(status);
 
	// status = fn_plugin.registerNode(
	// 	"rawfootPrint",
	// 	rawfootPrint::id,
	// 	&rawfootPrint::creator,
	// 	&rawfootPrint::initialize,
	// 	MPxNode::kLocatorNode,
	// 	&rawfootPrint::drawDbClassification
	// );
	// status = MHWRender::MDrawRegistry::registerDrawOverrideCreator(
	// 	rawfootPrint::drawDbClassification,
	// 	rawfootPrint::drawRegistrantId,
	// 	RawFootPrintDrawOverride::Creator
	// );

	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// Register a custom selection mask with priority 2 (same as locators by default).
	// MSelectionMask::registerSelectionType("footPrintSelection", 2);
	// status = MGlobal::executeCommand("selectType -byName \"footPrintSelection\" 1");


	if (MGlobal::mayaState() == MGlobal::kInteractive) {
		// Register callback to set selection priority on locators to 999
		setMelConfig(NULL);

		// afterNewCallbackId = MSceneMessage::addCallback(MSceneMessage::kAfterNew, setMelConfig, NULL, &status);
		// CHECK_MSTATUS_AND_RETURN_IT(status);
		// callbackIds.append(afterNewCallbackId);

		// afterOpenCallbackId = MSceneMessage::addCallback(MSceneMessage::kAfterOpen, setMelConfig, NULL, &status);
		// CHECK_MSTATUS_AND_RETURN_IT(status);
		// callbackIds.append(afterOpenCallbackId);
	
		// afterSaveSetMetaDataNodeCbId = MSceneMessage::addCallback(MSceneMessage::kBeforeSave, onSceneSaved, NULL, &status);
		// CHECK_MSTATUS_AND_RETURN_IT(status);
		// callbackIds.append(afterSaveSetMetaDataNodeCbId);

		// // Creates the maya main menu items
		// MGlobal::executePythonCommandOnIdle("from lunar.maya.resources.scripts.ctrlMainMenu import CtrlMainMenu");
		// MGlobal::executePythonCommandOnIdle("CtrlMainMenu().createMenuItems()");
	}

	return status;
}



MStatus uninitializePlugin(MObject obj) {
	MStatus status;
	MFnPlugin fn_plugin(obj);

	// MMessage::removeCallbacks(callbackIds);

	MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
		CtrlNode::type_drawdb,
		CtrlNode::type_drawid
	);
	fn_plugin.deregisterCommand(CtrlCommand::commandName);
	fn_plugin.deregisterNode(CtrlNode::type_id);

	fn_plugin.deregisterCommand(ComponentCmd::command_name);
	fn_plugin.deregisterNode(ComponentNode::type_id);

	// // Deregister TwistSolver
	// fn_plugin.deregisterNode(TwistSolver::typeId);

	// // Deregister Footroll Node
	// fn_plugin.deregisterNode(FootRollSolver::typeId);

	// // Deregister DenoiseNode
	// fn_plugin.deregisterNode(DenoiseNode::type_id);

	// Deregister MetaData command
	// fn_plugin.deregisterCommand(IkCommand::commandName);
	// // Deregister MetaData draw override
	// MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
	// 	MetaDataNode::drawDbClassification,
	// 	MetaDataNode::drawRegistrationId
	// );

	// // Deregister MetaDataNode
	// fn_plugin.deregisterNode(MetaDataNode::type_id);

	// Deregister IkCommand
	fn_plugin.deregisterCommand(IkCommand::commandName);

	// Deregister Ik2Solver
	fn_plugin.deregisterNode(Ik2bSolver::typeId);

	// Deregister SpaceSwitch
	fn_plugin.deregisterCommand(SpaceSwitchCmd::command_name);
	fn_plugin.deregisterNode(SpaceSwitchNode::type_id);


	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Deregister Controller node
	// status = fn_plugin.deregisterNode(Ctrl::typeId);
	// CHECK_MSTATUS_AND_RETURN_IT(status);

	// // Deletes the maya main menu items
	// if (MGlobal::mayaState() == MGlobal::kInteractive)
	// {
	// MGlobal::executePythonCommandOnIdle("CtrlMainMenu().deleteMenuItems()");
	// }


// 	MHWRender::MDrawRegistry::deregisterDrawOverrideCreator(
// 		rawfootPrint::drawDbClassification,
// 		rawfootPrint::drawRegistrantId
// 	);

// 	fn_plugin.deregisterNode(rawfootPrint::id);

// 	// Release DX resources
// #ifdef _WIN32
// 	{
// 		RawFootPrintDrawAgentDX& drawAgentRef = RawFootPrintDrawAgentDX::getDrawAgent();
// 		drawAgentRef.releaseDXResources();
// 	}
// #endif
// 	// Release GL Core resources
// 	{
// 		RawFootPrintDrawAgentCoreProfile& drawAgentRef = RawFootPrintDrawAgentCoreProfile::getDrawAgent();
// 		drawAgentRef.releaseCoreProfileResources();
// 	}

	return status;
}
