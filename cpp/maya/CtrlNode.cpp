#include "CtrlNode.h"
#include "ShapesDefinition.h"



//--------------------------------------------------------------------------------------
//
// Ctrl Transform Node implementation with standard viewport draw
//
//--------------------------------------------------------------------------------------


// MObject CtrlNode::attr_out_line_matrix	= {};
// MObject CtrlNode::inputSize            = {};
MObject CtrlNode::geometryChanging        = {};


// Class attributes
const MString CtrlNode::type_name 	= "ctrl";
const MTypeId CtrlNode::type_id 	  = 0x9000002;
const MString CtrlNode::type_drawdb	= "drawdb/geometry/animation/ctrl";
const MString CtrlNode::type_drawid	= "ctrlPlugin";

MObject CtrlNode::local_position, CtrlNode::local_positionX, CtrlNode::local_positionY, CtrlNode::local_positionZ;
MObject CtrlNode::local_rotate, CtrlNode::local_rotateX, CtrlNode::local_rotateY, CtrlNode::local_rotateZ;
MObject CtrlNode::local_scale, CtrlNode::local_scaleX, CtrlNode::local_scaleY, CtrlNode::local_scaleZ;

MObject CtrlNode::attr_shape_indx;
MObject CtrlNode::attr_line_width;

MObject CtrlNode::attr_in_draw_line;
Attribute CtrlNode::attr_in_line_matrix;
Attribute CtrlNode::attr_out_line_matrix;

MObject CtrlNode::attr_draw_solver_mode;
MObject CtrlNode::attr_solver_mode_size;
MObject CtrlNode::attr_solver_mode_positionX, CtrlNode::attr_solver_mode_positionY, CtrlNode::attr_solver_mode_positionZ, CtrlNode::attr_solver_mode_position;
MObject CtrlNode::attrInFkIk;
MObject CtrlNode::attr_component;


MStatus CtrlNode::initialize() {
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
	MStatus status;

	local_positionX = fn_num.create("localPositionX", "lpx", MFnNumericData::kFloat);
	local_positionY = fn_num.create("localPositionY", "lpy", MFnNumericData::kFloat);
	local_positionZ = fn_num.create("localPositionZ", "lpz", MFnNumericData::kFloat);
	local_position = fn_num.create("localPosition", "lp", local_positionX, local_positionY, local_positionZ);
	fn_num.setStorable(true);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	local_rotateX = fn_unit.create("localRotateX", "lrx", MFnUnitAttribute::kAngle);
	local_rotateY = fn_unit.create("localRotateY", "lry", MFnUnitAttribute::kAngle);
	local_rotateZ = fn_unit.create("localRotateZ", "lrz", MFnUnitAttribute::kAngle);
	local_rotate = fn_num.create("localRotate", "lr", local_rotateX, local_rotateY, local_rotateZ);
	fn_num.setStorable(true);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	local_scaleX = fn_num.create("localScaleX", "lsx", MFnNumericData::kFloat, 1.0);
	local_scaleY = fn_num.create("localScaleY", "lsy", MFnNumericData::kFloat, 1.0);
	local_scaleZ = fn_num.create("localScaleZ", "lsz", MFnNumericData::kFloat, 1.0);
	local_scale = fn_num.create("localScale", "ls", local_scaleX, local_scaleY, local_scaleZ);
	fn_num.setStorable(true);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	attr_shape_indx = fn_enum.create("shape", "shp");
	fn_enum.addField("Cube", 0);
	fn_enum.addField("Sphere", 1);
	fn_enum.addField("Cross", 2);
	fn_enum.addField("Diamond", 3);
	fn_enum.addField("Square", 4);
	fn_enum.addField("Circle", 5);
	fn_enum.addField("Locator", 6);
	fn_enum.addField("Line", 7);
	fn_enum.addField("None", 8);
	fn_enum.setKeyable(false);
	fn_enum.setStorable(true);
	fn_enum.setChannelBox(true);

	attr_line_width = fn_num.create("lineWidth", "lw", MFnNumericData::kFloat, 2.0);
	fn_num.setMin(0.5);
	fn_num.setDefault(2.0);
	fn_num.setMax(5);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	attr_in_draw_line = fn_num.create("drawLine", "dl", MFnNumericData::kBoolean, false);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	createAttribute(attr_in_line_matrix, "drawLineTo", DefaultValue<MMatrix>());

	attr_draw_solver_mode = fn_num.create("drawSolverMode", "dsm", MFnNumericData::kBoolean, false);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	attr_solver_mode_positionX = fn_num.create("solverModePositionX", "smpx", MFnNumericData::kDouble, 0.0);
	attr_solver_mode_positionY = fn_num.create("solverModePositionY", "smpy", MFnNumericData::kDouble, 0.0);
	attr_solver_mode_positionZ = fn_num.create("solverModePositionZ", "smpz", MFnNumericData::kDouble, 0.0);
	attr_solver_mode_position = fn_num.create("solverModePosition", "smp", attr_solver_mode_positionX, attr_solver_mode_positionY, attr_solver_mode_positionZ);
	fn_num.setStorable(true);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);

	attr_solver_mode_size = fn_num.create("solverModeSize", "sms", MFnNumericData::kInt, 12);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);
	fn_num.setMin(9);
	fn_num.setMax(24);

	attrInFkIk = fn_num.create("fkIk", "fkIk", MFnNumericData::kDouble);
	fn_num.setStorable(true);
	fn_num.setKeyable(false);
	fn_num.setChannelBox(true);
	fn_num.setSoftMin(0.0);
	fn_num.setSoftMax(100.0);

	attr_component = fn_mess.create("component", "component");
	fn_mess.setWritable(false);
	fn_mess.setReadable(true);

	geometryChanging = fn_num.create("geometryChanging", "gcg", MFnNumericData::kBoolean, true);
	fn_num.setStorable(false);
	fn_num.setHidden(true);
	fn_num.setConnectable(false);

	// Add attributes
	addAttributes(
		local_position, local_rotate, local_scale,
		attr_shape_indx, attr_line_width,
		attr_in_draw_line, attr_in_line_matrix,
		attr_draw_solver_mode,
		attr_solver_mode_size,
		attr_solver_mode_position,
		attrInFkIk,
		geometryChanging,
		attr_component
	);

	attributeAffects(CtrlNode::attr_in_line_matrix, CtrlNode::geometryChanging);

	return MS::kSuccess;
}


MStatus CtrlNode::setDependentsDirty(const MPlug& plug, MPlugArray& affectedPlugs) {
	/* Sets the relation between attributes and marks the specified plugs dirty.

	Args:
		plugBeingDirtied (&MPlug): Plug which is being set dirty by Maya.
		affectedPlugs (&MPlugArray): The programmer should add any plugs which they want to set dirty
			to this list.

	*/
	draw_line = MPlug(self_object, attr_in_draw_line).asBool();
	if (draw_line) MHWRender::MRenderer::setGeometryDrawDirty(self_object);

	// Update the shape attrs for editing / rig creation - draw - line has to be on for
	// performance optimizations
	if (MEvaluationManager::graphConstructionActive()) {
		if ( plug == attr_in_line_matrix
			|| plug == local_position
			|| plug == local_rotate
			|| plug == local_scale
			|| plug == attr_shape_indx
			|| plug == attr_line_width
			|| plug == attr_in_draw_line
			|| plug == attr_out_line_matrix
			|| plug == attr_draw_solver_mode
			|| plug == attr_solver_mode_size
			|| plug == attr_solver_mode_position
			|| plug == attrInFkIk
		) {affectedPlugs.append(MPlug(self_object, geometryChanging));}
	}

	return MS::kSuccess;
}


MStatus CtrlNode::compute(const MPlug& plug, MDataBlock& dataBlock) {
	// Check documentation in "class FootPrintNode" for descriptions about the attributes here
	if (plug == geometryChanging) {
		MDataHandle boolHandle = dataBlock.outputValue(geometryChanging);
		boolHandle.setBool(true);
		MHWRender::MRenderer::setGeometryDrawDirty(self_object);
	}	else {
		return MS::kUnknownParameter;
	}

	return MStatus::kSuccess;
}


// Called before this node is evaluated by Evaluation Manager
MStatus CtrlNode::postEvaluation(const MDGContext& context, const MEvaluationNode& evaluationNode, PostEvaluationType evalType) {
	// For cache restoration only
	// This method is responsible for fixing the 'geometryChanging' flag in cache restore frames
	// Because in cache store phase,
	// PopulateGeometry & Viewport-Caching happens before Evaluation-Cache store
	// The value of 'geometryChanging' will always be set to 'false' (it is already used by render)
	// Thus, we have to fix the geometryChanging attribute to the correct value 
	MStatus status;
	if (evalType == PostEvaluationEnum::kEvaluatedDirectly && evaluationNode.dirtyPlugExists(geometryChanging, &status) && status) {
		MDataBlock data = forceCache();
		MDataHandle boolHandle = data.outputValue(geometryChanging, &status);
		if (status != MStatus::kSuccess) return status;
		boolHandle.setBool(true);
		boolHandle.setClean();
	}

	return MPxTransform::postEvaluation(context, evaluationNode, evalType);
}


void CtrlNode::getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const {
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


void CtrlNode::postConstructor() {
	self_object=thisMObject();
	MDagPath::getAPathTo(self_object, self_path);

	MFnDependencyNode fn_this(self_object);

	// fn_this.findPlug("hideOnPlayback", false).setBool(1);
	// MPlug plug_scale = fn_this.findPlug("scale", false);
	// LMAttribute::lockAndHideAttr(plug_scale);
	fn_this.findPlug("shear", false).setLocked(1);
	fn_this.findPlug("rotateAxis", false).setLocked(1);
	// Set color
	fn_this.findPlug("overrideEnabled", false).setBool(1);
	fn_this.findPlug("overrideRGBColors", false).setBool(1);
	fn_this.findPlug("overrideColorR", false).setDouble(1.0);
	fn_this.findPlug("overrideColorG", false).setDouble(1.0);
	fn_this.findPlug("overrideColorB", false).setDouble(0.25);

	draw_line = MPlug(self_object, CtrlNode::attr_in_draw_line).asBool();
}


MBoundingBox CtrlNode::boundingBox() const {
	// Get the size
	CtrlUserData data;
	data.get_plugs(self_object);
	data.get_bbox(self_object, self_path, data.mat_local);

	return data.bbox;
}


void CtrlUserData::get_plugs(const MObject& object) {
	/* Get all the necessary data from the attributes of the locator.

	Args:
		obj (MObject &): Object passed to query the attrbutes from

	*/
	float tx = MPlug(object, CtrlNode::local_positionX).asFloat();
	float ty = MPlug(object, CtrlNode::local_positionY).asFloat();
	float tz = MPlug(object, CtrlNode::local_positionZ).asFloat();

	float rx = MPlug(object, CtrlNode::local_rotateX).asFloat();
	float ry = MPlug(object, CtrlNode::local_rotateY).asFloat();
	float rz = MPlug(object, CtrlNode::local_rotateZ).asFloat();

	float sx = MPlug(object, CtrlNode::local_scaleX).asFloat();
	float sy = MPlug(object, CtrlNode::local_scaleY).asFloat();
	float sz = MPlug(object, CtrlNode::local_scaleZ).asFloat();

	// MEulerRotation rot_euler(rx, ry, rz);
	this->mat_local = MEulerRotation(rx, ry, rz).asMatrix();
	this->mat_local[3][0] = tx; this->mat_local[3][1] = ty;	this->mat_local[3][2] = tz;
	this->mat_local[0][0] *= sx; this->mat_local[0][1] *= sx;	this->mat_local[0][2] *= sx;
	this->mat_local[1][0] *= sy; this->mat_local[1][1] *= sy;	this->mat_local[1][2] *= sy;
	this->mat_local[2][0] *= sz; this->mat_local[2][1] *= sz;	this->mat_local[2][2] *= sz;

	MFnDependencyNode fn_object(object);

	bDrawline = MPlug(object, CtrlNode::attr_in_draw_line).asBool();
	line_width = MPlug(object, CtrlNode::attr_line_width).asFloat();
	shape_indx = MPlug(object, CtrlNode::attr_shape_indx).asShort();
	solver_mode_size = MPlug(object, CtrlNode::attr_solver_mode_size).asInt();

	mat_pv = MDataHandle(MPlug(object, CtrlNode::attr_in_line_matrix).asMDataHandle()).asMatrix();
	pos_draw_pv_to = MPoint(mat_pv[3][0], mat_pv[3][1], mat_pv[3][2]);
}


void CtrlUserData::get_bbox(const MObject& object, const MDagPath& dp_object, MMatrix matrix) {
	/* Gets the bounding box from the shapesDefinition.h file

	Args:
		obj (MObject &): Object passed to query the attrbutes from
		matrix (MMatrix): Matrix used to transform the bounding box

	*/
	shape_indx = MPlug(object, CtrlNode::attr_shape_indx).asShort();
	// Cube
	if (shape_indx == 0) {
		this->bbox = MBoundingBox(
			MPoint(cubeBB[0][0], cubeBB[0][1], cubeBB[0][2]),	MPoint(cubeBB[1][0], cubeBB[1][1], cubeBB[1][2])
		);
	// Sphere
	} else if (shape_indx == 1) {
		this->bbox = MBoundingBox(
			MPoint(sphereBB[0][0], sphereBB[0][1], sphereBB[0][2]),	MPoint(sphereBB[1][0], sphereBB[1][1], sphereBB[1][2])
		);
	// Cross
	} else if (shape_indx == 2) {
		this->bbox = MBoundingBox(
			MPoint(crossBB[0][0], crossBB[0][1], crossBB[0][2]), MPoint(crossBB[1][0], crossBB[1][1], crossBB[1][2])
		);
	// Diamond
	} else if (shape_indx == 3) {
		this->bbox = MBoundingBox(
			MPoint(diamondBB[0][0], diamondBB[0][1], diamondBB[0][2]), MPoint(diamondBB[1][0], diamondBB[1][1], diamondBB[1][2])
		);
	// Square
	} else if (shape_indx == 4) {
		this->bbox = MBoundingBox(
			MPoint(squareBB[0][0], squareBB[0][1], squareBB[0][2]),	MPoint(squareBB[1][0], squareBB[1][1], squareBB[1][2])
		);
	// Circle
	} else if (shape_indx == 5) {
		this->bbox = MBoundingBox(
			MPoint(circleBB[0][0], circleBB[0][1], circleBB[0][2]),	MPoint(circleBB[1][0], circleBB[1][1], circleBB[1][2])
		);

	// Locator
	} else if (shape_indx == 6) {
		this->bbox = MBoundingBox(
			MPoint(nullBB[0][0], nullBB[0][1], nullBB[0][2]),	MPoint(nullBB[1][0], nullBB[1][1], nullBB[1][2])
		);

	// Line
	} else if (shape_indx == 7) {
		this->bbox = MBoundingBox(
			MPoint(lineBB[0][0], lineBB[0][1], lineBB[0][2]), MPoint(lineBB[1][0], lineBB[1][1], lineBB[1][2])
		);
	// None
	} else if (shape_indx == 8) {
		this->bbox = MBoundingBox(MPoint(0.0, 0.0, 0.0), MPoint(0.0, 0.0, 0.0)
		);
	}
	this->bbox.transformUsing(matrix);
	this->bbox.expand(this->pos_draw_pv_to);
}


void CtrlUserData::get_shape(const MObject& object, const MDagPath& dp_object, MMatrix matrix) {
	/* Get the points for each line and triangle used for drawing the shape.

	Do not reorder the triangle append order since it will flip normals.

	Args:
		obj (MObject &): Object passed to query the attrbutes from
		matrix (MMatrix): Matrix used to transform the line points

	*/
	MStatus status;

	shape_indx = MPlug(object, CtrlNode::attr_shape_indx).asShort();

	this->list_vertecies.clear();
	this->list_lines.clear();

	if (shape_indx == 0) {  // Cube
		for (int i=0; i<cubeCount; i++) {
			list_vertecies.append(
				MPoint(listPointsCube[i][0], listPointsCube[i][1], listPointsCube[i][2]) * matrix
			);
		}
		// Top Quad
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[0]);
		// Side Lines
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[7]);
		// Bottom Quad
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[4]);

	} else if (shape_indx == 1) {	  // Sphere
		for (int i=0; i<sphereCount; i++) {
			list_vertecies.append(
				MPoint(listPointsSphere[i][0], listPointsSphere[i][1], listPointsSphere[i][2]) * matrix
			);
		}
		// Top square
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[0]);
		// Top lines
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[6]);

		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[8]);

		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[8]);
		list_lines.append(list_vertecies[8]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[10]);

		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[4]);
		// Side Lines
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[12]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[8]);
		list_lines.append(list_vertecies[16]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[19]);
		// Bottom lines
		list_lines.append(list_vertecies[20]);
		list_lines.append(list_vertecies[12]);
		list_lines.append(list_vertecies[12]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[20]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[14]);

		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[16]);

		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[16]);
		list_lines.append(list_vertecies[16]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[18]);

		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[12]);

		// Bottom square
		list_lines.append(list_vertecies[20]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[20]);

	} else if (shape_indx == 2) {  	// Cross
		for (int i=0; i<crossCount; i++) {
			list_vertecies.append(
				MPoint(listPointsCross[i][0], listPointsCross[i][1], listPointsCross[i][2]) * matrix
			);
		}
		// Base upper square
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[0]);
		// Base side lines
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[7]);
		// Base bottom square
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[4]);
		// Top pillar
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[8]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[8]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[9]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[10]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[11]);
		list_lines.append(list_vertecies[8]);
		// Right pillar
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[12]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[12]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[13]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[14]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[15]);
		list_lines.append(list_vertecies[12]);
		// Back pillar
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[16]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[16]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[17]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[18]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[19]);
		list_lines.append(list_vertecies[16]);
		// Left pillar
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[20]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[20]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[21]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[22]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[23]);
		list_lines.append(list_vertecies[20]);
		// Front pillar
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[24]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[25]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[26]);
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[27]);
		list_lines.append(list_vertecies[24]);
		list_lines.append(list_vertecies[25]);
		list_lines.append(list_vertecies[25]);
		list_lines.append(list_vertecies[26]);
		list_lines.append(list_vertecies[26]);
		list_lines.append(list_vertecies[27]);
		list_lines.append(list_vertecies[27]);
		list_lines.append(list_vertecies[24]);
		// Bottom pillar
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[28]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[29]);
		list_lines.append(list_vertecies[6]);
		list_lines.append(list_vertecies[30]);
		list_lines.append(list_vertecies[7]);
		list_lines.append(list_vertecies[31]);
		list_lines.append(list_vertecies[28]);
		list_lines.append(list_vertecies[29]);
		list_lines.append(list_vertecies[29]);
		list_lines.append(list_vertecies[30]);
		list_lines.append(list_vertecies[30]);
		list_lines.append(list_vertecies[31]);
		list_lines.append(list_vertecies[31]);
		list_lines.append(list_vertecies[28]);
	
	} else if (shape_indx == 3) {  	// Diamond
		for (int i=0; i<diamondCount; i++) {
			list_vertecies.append(
				MPoint(listPointsDiamond[i][0], listPointsDiamond[i][1], listPointsDiamond[i][2]) * matrix
			);
		}
		// Top lines 
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[4]);
		// Planar lines
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[4]);
		list_lines.append(list_vertecies[1]);
		// Bottom lines
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[5]);
		list_lines.append(list_vertecies[4]);

	} else if (shape_indx == 4) {   // Square
		for (int i=0; i<squareCount; i++) {
			list_vertecies.append(
				MPoint(listPointsSquare[i][0], listPointsSquare[i][1], listPointsSquare[i][2]) * matrix
			);
		}
		list_lines.append(list_vertecies[0]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[1]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[2]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[3]);
		list_lines.append(list_vertecies[0]);

	} else if (shape_indx == 5) {   // Circle
		for (int i=0; i<circleCount; i++) {
			list_vertecies.append(
				MPoint(listPointsCircle[i][0], listPointsCircle[i][1], listPointsCircle[i][2]) * matrix
			);
		}
		for (int i=1; i<=circleCount-2; i++) {
			list_lines.append(list_vertecies[i]);
			list_lines.append(list_vertecies[i+1]);
		}

	} else if (shape_indx == 6) {  	// Locator
		for (int i=0; i<nullCount; i++) {
			list_lines.append(MPoint(listPointsNull[i][0], listPointsNull[i][1], listPointsNull[i][2]) * matrix);
		}
	} else if (shape_indx == 7) {  	// Line
		list_lines.append(MPoint(listPointsLine[0][0], listPointsLine[0][1], listPointsLine[0][2]) * matrix);
		list_lines.append(MPoint(listPointsLine[1][0], listPointsLine[1][1], listPointsLine[1][2]) * matrix);
	}

	// if (showOrientation) {
	// 	currentInd++;
	// 	MPointArray& linesPointsCenterOrientation = this->list_lines[currentInd];
	// 	linesPointsCenterOrientation.clear();
	// 	for (int i = 0; i < lookAtCount; i++)
	// 		linesPointsCenterOrientation.append(MPoint(listLinesLookat[i][0] * centerScale, listLinesLookat[i][1] * centerScale, listLinesLookat[i][2] * centerScale));
	// }

	// Draw line for pole vectors
	if (bDrawline) { 
		// MMatrix matDrawLineTo = MDataHandle(MPlug(object, CtrlNode::attr_in_line_matrix).asMDataHandle()).asMatrix();
		list_lines.append(MPoint() * matrix);
		list_lines.append(MPoint(pos_draw_pv_to[0], pos_draw_pv_to[1], pos_draw_pv_to[2]) * dp_object.inclusiveMatrixInverse());
	}
}


void CtrlUserData::get_text(const MObject& object) {
	draw_solver_mode = MPlug(object, CtrlNode::attr_draw_solver_mode).asBool();
	pos_solver_mode = MPoint(
		MPlug(object, CtrlNode::attr_solver_mode_positionX).asFloat(),
		MPlug(object, CtrlNode::attr_solver_mode_positionY).asFloat(),
		MPlug(object, CtrlNode::attr_solver_mode_positionZ).asFloat()
	);
	fkIk = MPlug(object, CtrlNode::attrInFkIk).asDouble();
	if (fkIk == 0.0) {
		str_solver_mode = MString("Fk");
	} else if (fkIk > 0.0 && fkIk < 100.0) {
		MString strFkIk = LMText::doublePrecision(MPlug(object, CtrlNode::attrInFkIk).asDouble(), 0).c_str();
		str_solver_mode = MString("FkIk " + strFkIk);
	} else if (fkIk == 100.0) {
		str_solver_mode = MString("Ik");
	}
}


void CtrlDrawOverride::OnModelEditorChanged(void *clientData) {
	// Mark the node as being dirty so that it can update on display appearance
	// switch among wireframe and shaded.
	CtrlDrawOverride *ovr = static_cast<CtrlDrawOverride*>(clientData);
	if (ovr && ovr->ptrCtrlNode) {MHWRender::MRenderer::setGeometryDrawDirty(ovr->ptrCtrlNode->thisMObject());}
}


MBoundingBox CtrlDrawOverride::boundingBox(const MDagPath& objPath, const MDagPath& cameraPath) const {
	/* Called by Maya whenever the bounding box of the drawable object is needed.

	This method should return the object space bounding box for the object to be drawn.
	Note that this method will not be called if the isBounded() method returns a value of
	false.
	Default implementation returns a huge bounding box which will never cull the object.

	Args:
		objPath (MDagPath&): The path to the object being drawn
		cameraPath (MDagPath&): The path to the camera that is being used to draw

	Returns:
		MBoundingBox: The object space bounding box of object drawn in the draw callback

	*/
	CtrlUserData data;
	MObject node = objPath.node();

	data.get_plugs(node);
	data.get_bbox(node, objPath, data.mat_local);

	return data.bbox;
}


MUserData* CtrlDrawOverride::prepareForDraw(const MDagPath& objPath, const MDagPath& cameraPath, const MHWRender::MFrameContext& frameContext, MUserData* oldData) {
	/* Called by Maya whenever the object is dirty and needs to update for draw.

	Any data needed from the Maya dependency graph must be retrieved and cached in this
	stage. It is invalid to pull data from the Maya dependency graph in the draw callback
	method and Maya may become unstable if that is attempted.

	Implementors may allow Maya to handle the data caching by returning a pointer to the
	data from this method. The pointer must be to a class derived from MUserData. This
	same pointer will be passed to the draw callback. On subsequent draws, the pointer
	will also be passed back into this method so that the data may be modified and reused
	instead of reallocated. If a different pointer is returned Maya will delete the old
	data. If the cache should not be maintained between draws, set the delete after use
	flag on the user data and Maya will delete it in a certain stage depending on whether
	the draw callback exists. If exists, the user data will be deleted after both
	addUIDrawables() and the draw callback are called; otherwise, the user data will be
	deleted immediately after addUIDrawables() is called. In all cases, the lifetime and
	ownership of the user data is handled by Maya and the user cshould not try to delete
	the data themselves. Data caching occurs per-instance of the associated DAG object.
	The lifetime of the user data can be longer than the associated node, instance or
	draw override. Due to internal caching, the user data can be deleted after an
	arbitrary long time. One should therefore be careful to not access stale objects from
	the user data destructor. If it is not desirable to allow Maya to handle data caching,
	simply return NULL in this method and ignore the user data parameter in the draw
	callback method.

	Args:
		objPath (MDagPath&): The path to the object being drawn
		cameraPath (MDagPath&): The path to the camera that is being used to draw
		frameContext (MHWRender::MFrameContext&): Frame level context information
		oldData (MUserData*) Data cached by the previous draw of the instance

	Returns:
		MUserData*: Pointer to data to be passed to the draw callback method

	*/
	MStatus status;

	CtrlUserData* data = dynamic_cast<CtrlUserData*>(oldData);
	MObject object = objPath.node();

	if (!data) {data=new CtrlUserData;}

	data->get_plugs(object);
	data->get_shape(object, objPath, data->mat_local);
	data->get_text(object);

	data->col_wireframe = MHWRender::MGeometryUtilities::wireframeColor(objPath);

	// If XRay Joints Draw in XRay Mode
	// if (frameContext.getDisplayStyle() & MHWRender::MFrameContext::kXrayJoint) {data->DrawInXray = true;}
	// else {data->DrawInXray = false;}

	switch (MHWRender::MGeometryUtilities::displayStatus(objPath)) {
		case MHWRender::kLead:
		case MHWRender::kActive:
		case MHWRender::kHilite:
		case MHWRender::kActiveComponent:
			data->prio_depth = MHWRender::MRenderItem::sActiveWireDepthPriority;
			break;
		default:
			data->prio_depth = MHWRender::MRenderItem::sDormantFilledDepthPriority;
			break;
	}

	return data;
}


void CtrlDrawOverride::addUIDrawables(const MDagPath& objPath, MHWRender::MUIDrawManager& drawManager, const MHWRender::MFrameContext& frameContext, const MUserData* data) {
	/* Provides access to the MUIDrawManager, which can be used to queue up operations
	to draw simple UI shapes like lines, circles, text, etc.

	This method will only be called when this override is dirty and its hasUIDrawables()
	is overridden to return true. This method is called after prepareForDraw() and carries
	the same restrictions on the sorts of operations it can perform.

	Args:
		objPath (MDagPath&): The path to the object being drawn
		drawManager (MHWRender::MUIDrawManager&): The UI draw manager, it can be used to
			draw some simple geometry including text
		frameContext (MHWRender::MFrameContext&): Frame level context information
		data (MUserData*) Data cached by prepareForDraw()

	*/
	CtrlUserData* pTransformData = (CtrlUserData*)data;
	if (!pTransformData) {return;}

	// Define selectability
	if (pTransformData->shape_indx == 8) {
		drawManager.beginDrawable(MHWRender::MUIDrawManager::kNonSelectable);
	} else {
		drawManager.beginDrawable();
	}

	drawManager.setDepthPriority(pTransformData->prio_depth);

	// // If XRay Joints Draw in XRay Mode
	// if (pCtrlData->DrawInXray) {drawManager.beginDrawInXray();}
	
	drawManager.setColor(pTransformData->col_wireframe);
	drawManager.setLineWidth(pTransformData->line_width);

	drawManager.mesh(MHWRender::MUIDrawManager::kLines, pTransformData->list_lines);
	if (pTransformData->bDrawline) {drawManager.mesh(MHWRender::MUIDrawManager::kLines, pTransformData->list_lines);}

	// Fk Ik State
	if (pTransformData->draw_solver_mode) {
		drawManager.setFontSize(pTransformData->solver_mode_size);
		drawManager.text(pTransformData->pos_solver_mode, pTransformData->str_solver_mode, drawManager.kCenter);
	}

	// End drawable
	// if (pCtrlData->DrawInXray) {drawManager.endDrawInXray();}

	drawManager.endDrawable();
}

