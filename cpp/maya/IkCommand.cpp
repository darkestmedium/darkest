#include "Ik2bSolver.h"
#include "IkCommand.h"



// Public Data
const char* IkCommand::commandName = "ik";

// Command's Flags
const char* IkCommand::nameFlagShort = "-n";
const char* IkCommand::nameFlagLong = "-name";

// Fk Flags
const char* IkCommand::fkStartFlagShort = "-fks";
const char* IkCommand::fkStartFlagLong = "-fkStart";
const char* IkCommand::fkMidFlagShort = "-fkm";
const char* IkCommand::fkMidFlagLong = "-fkMid";
const char* IkCommand::fkEndFlagShort = "-fke";
const char* IkCommand::fkEndFlagLong = "-fkEnd";
// Ik Flags
const char* IkCommand::ikStartFlagShort = "-iks";
const char* IkCommand::ikStartFlagLong = "-ikStart";
const char* IkCommand::ikMidFlagShort = "-ikm";
const char* IkCommand::ikMidFlagLong = "-ikMid";
const char* IkCommand::ikEndFlagShort = "-ike";
const char* IkCommand::ikEndFlagLong = "-ikEnd";
const char* IkCommand::ikHandleFlagShort = "-ikh";
const char* IkCommand::ikHandleFlagLong = "-ikHandle";
const char* IkCommand::poleVectorFlagShort = "-pv";
const char* IkCommand::poleVectorFlagLong = "-poleVector";
// Out Flags
const char* IkCommand::outStartFlagShort = "-os";
const char* IkCommand::outStartFlagLong = "-outStart";
const char* IkCommand::outMidFlagShort = "-om";
const char* IkCommand::outMidFlagLong = "-outMid";
const char* IkCommand::outEndFlagShort = "-oe";
const char* IkCommand::outEndFlagLong = "-outEnd";

const char* IkCommand::modeFlagShort = "-mod";
const char* IkCommand::modeFlagLong = "-mode";

// Local position flags
const char* IkCommand::helpFlagShort = "-h";
const char* IkCommand::helpFlagLong = "-help";



MSyntax IkCommand::syntaxCreator() {
	/* Creates the command's syntax object and returns it.

	Returns:
		syntax (MSyntax): Command's syntax object

	*/
	MSyntax sytnax;

	// Main flags
	sytnax.addFlag(nameFlagShort, nameFlagLong, MSyntax::kString);

	// Fk Flags
	sytnax.addFlag(fkStartFlagShort, fkStartFlagLong, MSyntax::kString);
	sytnax.addFlag(fkMidFlagShort, fkMidFlagLong, MSyntax::kString);
	sytnax.addFlag(fkEndFlagShort, fkEndFlagLong, MSyntax::kString);
	// Ik Flags
	sytnax.addFlag(ikStartFlagShort, ikStartFlagLong, MSyntax::kString);
	sytnax.addFlag(ikMidFlagShort, ikMidFlagLong, MSyntax::kString);
	sytnax.addFlag(ikEndFlagShort, ikEndFlagLong, MSyntax::kString);
	sytnax.addFlag(ikHandleFlagShort, ikHandleFlagLong, MSyntax::kString);
	sytnax.addFlag(poleVectorFlagShort, poleVectorFlagLong, MSyntax::kString);
	// Out Flags
	sytnax.addFlag(outStartFlagShort, outStartFlagLong, MSyntax::kString);
	sytnax.addFlag(outMidFlagShort, outMidFlagLong, MSyntax::kString);
	sytnax.addFlag(outEndFlagShort, outEndFlagLong, MSyntax::kString);

	sytnax.addFlag(modeFlagShort, modeFlagLong, MSyntax::kString);

	// Help Flags	
	sytnax.addFlag(helpFlagShort, helpFlagLong, MSyntax::kBoolean);

	sytnax.useSelectionAsDefault(false);

	return sytnax;
}


MStatus IkCommand::parseArguments(const MArgList& argList) {
	/* Parses the commands's flag arguments.

	Args:
		argList (MArglist): List of arguments passed to the command.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	MArgDatabase argData(syntax(), argList);

	// Display Help
	if (argData.isFlagSet(helpFlagShort))	{
		command = kCommandHelp;
		MString helpStr;
		helpStr += "Flags:\n";
		helpStr += "   -n     -name                 String     Name of the ik solver node to be created.\n";
		helpStr += "   -fks   -fkStart              String     Name of the fk start transform input.\n";
		helpStr += "   -fkm   -fkMid                String     Name of the fk mid transform input.\n";
		helpStr += "   -fke   -fkEnd                String     Name of the fk end transform input.\n";
		helpStr += "   -iks   -ikStart              String     Name of the ik start transform input.\n";
		helpStr += "   -ikm   -ikMid                String     Name of the ik mid transform input.\n";
		helpStr += "   -ike   -ikEnd                String     Name of the ik end transform input.\n";
		helpStr += "   -ikh   -ikHandle             String     Name of the ik handle transform input.\n";
		helpStr += "   -pv    -poleVector           String     Name of the pole vector transform input (optional).\n";
		helpStr += "   -os    -outStart             String     Name of the start joint input.\n";
		helpStr += "   -om    -outMid               String     Name of the mid joint input.\n";
		helpStr += "   -oe    -outEnd               String     Name of the end joint input.\n";
		helpStr += "   -mod   -mode                 String     Solver mode 'fk' or 'ik'.\n";
		helpStr += "   -h     -help                 N/A        Display this text.\n";
		MGlobal::displayInfo(helpStr);
		return MS::kSuccess;
	}

	// Name Flag
	if (argData.isFlagSet(nameFlagShort))	{
		name = argData.flagArgumentString(nameFlagShort, 0, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
	}

	// Fk Start Flag
	if (argData.isFlagSet(fkStartFlagShort)) {
		MDagPath dpFkStart;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(fkStartFlagShort, 0, &status), dpFkStart);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnFkStart.setObject(dpFkStart);
	} else {
		MGlobal::displayError("fkStart flag is required.");
		return MS::kFailure;
	}

	// Fk Mid Flag
	if (argData.isFlagSet(fkMidFlagShort)) {
		MDagPath dpFkMid;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(fkMidFlagShort, 0, &status), dpFkMid);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnFkMid.setObject(dpFkMid);
	} else {
		MGlobal::displayError("fkMid flag is required.");
		return MS::kFailure;
	}

	// Fk End Flag
	if (argData.isFlagSet(fkEndFlagShort)) {
		MDagPath dpFkEnd;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(fkEndFlagShort, 0, &status), dpFkEnd);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnFkEnd.setObject(dpFkEnd);
	} else {
		MGlobal::displayError("fkEnd flag is required.");
		return MS::kFailure;
	}

	// Ik Start Flag
	if (argData.isFlagSet(ikStartFlagShort)) {
		MDagPath dpIkStart;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(ikStartFlagShort, 0, &status), dpIkStart);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnIkStart.setObject(dpIkStart);
	} else {
		MGlobal::displayError("ikStart flag is required.");
		return MS::kFailure;
	}

	// Ik Mid Flag
	if (argData.isFlagSet(ikMidFlagShort)) {
		MDagPath dpIkMid;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(ikMidFlagShort, 0, &status), dpIkMid);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnIkMid.setObject(dpIkMid);
	} else {
		MGlobal::displayError("ikkMid flag is required.");
		return MS::kFailure;
	}

	// Ik End Flag
	if (argData.isFlagSet(ikEndFlagShort)) {
		MDagPath dpIkEnd;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(ikEndFlagShort, 0, &status), dpIkEnd);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnIkEnd.setObject(dpIkEnd);
	} else {
		MGlobal::displayError("ikEnd flag is required.");
		return MS::kFailure;
	}

	// Ik Handle Flag
	if (argData.isFlagSet(ikHandleFlagShort)) {
		MDagPath dpIkHandle;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(ikHandleFlagShort, 0, &status), dpIkHandle);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnIkHandle.setObject(dpIkHandle);
	} else {
		MGlobal::displayError("ikHandle flag is required.");
		return MS::kFailure;
	}

	// PoleVector Flag
	if (argData.isFlagSet(poleVectorFlagShort)) {
		MString strPoleVector = argData.flagArgumentString(poleVectorFlagShort, 0, &status);
		if (strPoleVector != "") {
			status = LMObject::getDagPathFromString(strPoleVector, dpPoleVector);
			CHECK_MSTATUS_AND_RETURN_IT(status);
			fnPoleVector.setObject(dpPoleVector);
			bIsPoleVectorSet = true;
		}
	} else {
		fnPoleVector.setObject(MObject::kNullObj);
		bIsPoleVectorSet = false;
	}

	// Joint Start Flag
	if (argData.isFlagSet(outStartFlagShort)) {
		MDagPath dpJntStart;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(outStartFlagShort, 0, &status), dpJntStart);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnJntStart.setObject(dpJntStart);
	} else {
		MGlobal::displayError("outStart flag is required.");
		return MS::kFailure;
	}

	// Joint Mid Flag
	if (argData.isFlagSet(outMidFlagShort)) {
		MDagPath dpJntMid;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(outMidFlagShort, 0, &status), dpJntMid);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnJntMid.setObject(dpJntMid);
	} else {
		MGlobal::displayError("outMid flag is required.");
		return MS::kFailure;
	}

	// Joint End Flag
	if (argData.isFlagSet(outEndFlagShort)) {
		MDagPath dpJntEnd;
		status = LMObject::getDagPathFromString(argData.flagArgumentString(outEndFlagShort, 0, &status), dpJntEnd);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		fnJntEnd.setObject(dpJntEnd);
	} else {
		MGlobal::displayError("outEnd flag is required.");
		return MS::kFailure;
	}

	// Mode Flag
	if (argData.isFlagSet(modeFlagShort)) {
		MString strMode = argData.flagArgumentString(modeFlagShort, 0, &status);
		CHECK_MSTATUS_AND_RETURN_IT(status);
		if (strMode == "fk") {
			mode = 0.0;
		}
		if (strMode == "ik") {
			mode = 100.0;
		}
	}

	return MS::kSuccess;
}


MStatus IkCommand::doIt(const MArgList& argList) {
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
		// Solver node creation
		objIk2bSolver = modDg.createNode(Ik2bSolver::typeName);
		if (name == Ik2bSolver::typeName)	{
			modDg.renameNode(objIk2bSolver, name);
		}	else {
			modDg.renameNode(objIk2bSolver, name + Ik2bSolver::typeName);
		}
		// Get function sets
		fnIk2bSolver.setObject(objIk2bSolver);

		// Get Fk input plugs
		MPlug plugOutFkStartWorldMatrix0 = fnFkStart.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutFkMidWorldMatrix0 = fnFkMid.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutFkEndWorldMatrix0 = fnFkEnd.findPlug("worldMatrix", false).elementByLogicalIndex(0);

		// Get Ik input plugs
		MPlug plugOutIkStartWorldMatrix0 = fnIkStart.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutIkMidWorldMatrix0 = fnIkMid.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutIkEndWorldMatrix0 = fnIkEnd.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutIkHandleWorldMatrix0 = fnIkHandle.findPlug("worldMatrix", false).elementByLogicalIndex(0);

		// Get Out input plugs
		MPlug plugOutJntStartWorldMatrix0 = fnJntStart.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutJntMidWorldMatrix0 = fnJntMid.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		MPlug plugOutJntEndWorldMatrix0 = fnJntEnd.findPlug("worldMatrix", false).elementByLogicalIndex(0);
		
		// Get solver input plugs
		MPlug plugInFkStart = fnIk2bSolver.findPlug("fkStart", false);
		MPlug plugInFkMid = fnIk2bSolver.findPlug("fkMid", false);
		MPlug plugInFkEnd = fnIk2bSolver.findPlug("fkEnd", false);
	
		MPlug plugInIkStart = fnIk2bSolver.findPlug("ikStart", false);
		MPlug plugInIkMid = fnIk2bSolver.findPlug("ikMid", false);
		MPlug plugInIkEnd = fnIk2bSolver.findPlug("ikEnd", false);
	
		MPlug plugInIkHandle = fnIk2bSolver.findPlug("ikHandle", false);
		MPlug plugInPoleVector = fnIk2bSolver.findPlug("poleVector", false);

		MPlug plugInOutStart = fnIk2bSolver.findPlug("outStart", false);
		MPlug plugInOutMid = fnIk2bSolver.findPlug("outMid", false);
		MPlug plugInOutEnd = fnIk2bSolver.findPlug("outEnd", false);
	
		// Connect matrix plugs
		modDg.connect(plugOutFkStartWorldMatrix0, plugInFkStart);
		modDg.connect(plugOutFkMidWorldMatrix0, plugInFkMid);
		modDg.connect(plugOutFkEndWorldMatrix0, plugInFkEnd);
	
		modDg.connect(plugOutIkStartWorldMatrix0, plugInIkStart);
		modDg.connect(plugOutIkMidWorldMatrix0, plugInIkMid);
		modDg.connect(plugOutIkEndWorldMatrix0, plugInIkEnd);
		modDg.connect(plugOutIkHandleWorldMatrix0, plugInIkHandle);

		modDg.connect(plugOutJntStartWorldMatrix0, plugInOutStart);
		modDg.connect(plugOutJntMidWorldMatrix0, plugInOutMid);
		modDg.connect(plugOutJntEndWorldMatrix0, plugInOutEnd);

		// Get rotation output plugs
		MPlug plugOutRotStart = fnIk2bSolver.findPlug("rotateStart", false);
		MPlug plugOutRotMid = fnIk2bSolver.findPlug("rotateMid", false);
		MPlug plugOutRotEnd = fnIk2bSolver.findPlug("rotateEnd", false);

		// Get joints rotation plugs
		MPlug plugRotStart = fnJntStart.findPlug("rotate", false);
		MPlug plugRotMid = fnJntMid.findPlug("rotate", false);
		MPlug plugRotEnd = fnJntEnd.findPlug("rotate", false);

		modDg.connect(plugOutRotStart, plugRotStart);
		modDg.connect(plugOutRotMid, plugRotMid);
		modDg.connect(plugOutRotEnd, plugRotEnd);
	
		MPlug plugOutUpdate = fnIk2bSolver.findPlug("update", false);
		MPlug plugRotPivotOutX = fnJntStart.findPlug("rotatePivotX", false);
		MPlug plugRotPivotIkX = fnIkStart.findPlug("rotatePivotX", false);
		modDg.connect(plugOutUpdate, plugRotPivotOutX);
		// modDg.connect(plugOutUpdate, plugRotPivotIkX);

		// Pole vector plugs
		if (bIsPoleVectorSet) {
			MPlug plugPoleVectorTranslate = fnPoleVector.findPlug("translate", false);

			// Get poleVectorShape
			dpPoleVector.extendToShape();
			MFnDependencyNode fnPoleVectorShape = dpPoleVector.node();
			MPlug plugPoleVectorDrawLineTo = fnPoleVectorShape.findPlug("drawLineTo", false);
			
			modDg.connect(plugPoleVectorTranslate, plugInPoleVector);
			// modDg.connect(plugOutUpdate, plugPoleVectorRotPivot);
			// modDg.connect(plugOutFkMidWorldMatrix0, plugPoleVectorDrawLineTo);
		}

		MPlug plugMode = fnIk2bSolver.findPlug("fkIk", false);
		plugMode.setValue(mode);

	}

	return redoIt();
}


MStatus IkCommand::redoIt() {
	/* Command's redoIt method.

	This method should do the actual work of the command based on the internal class data only.	Internal class data should be set in the doIt method.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/

	// Command create mode
	if (command == kCommandCreate) {
		MStatus status;

		status = modDg.doIt();
		CHECK_MSTATUS_AND_RETURN_IT(status);

		if (!bIsPoleVectorSet) {
			MPlug plugInPoleVectorX = fnIk2bSolver.findPlug("poleVectorX", false);
			MPlug plugInPoleVectorY = fnIk2bSolver.findPlug("poleVectorY", false);
			MPlug plugInPoleVectorZ = fnIk2bSolver.findPlug("poleVectorZ", false);

			MVector posFkStart = fnFkStart.rotatePivot(MSpace::kWorld);
			MVector posFkMid = fnFkMid.rotatePivot(MSpace::kWorld);
			MVector posFkEnd = fnFkEnd.rotatePivot(MSpace::kWorld);
			MVector posPoleVector = LMRigUtils::getPvPosition(posFkStart, posFkMid, posFkEnd, "local");

			plugInPoleVectorX.setValue(posPoleVector.x);
			plugInPoleVectorY.setValue(posPoleVector.y);
			plugInPoleVectorZ.setValue(posPoleVector.z);
		}

		// Sets command's output result in mel / python
		clearResult();
		appendToResult(fnIk2bSolver.name());
	}

	return MS::kSuccess;
}


MStatus IkCommand::undoIt() {
	/* Command's undoIt method.

	This method should undo the work done by the redoIt method based on the internal class data only.

	Returns:
		status code (MStatus): kSuccess if the command was successful, kFailure if an error occured
			during the command.

	*/
	MStatus status;

	// Restore the initial state
	status = modDg.undoIt();
	CHECK_MSTATUS_AND_RETURN_IT(status);

	return MS::kSuccess;
}

