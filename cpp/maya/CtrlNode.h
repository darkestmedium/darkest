// ==========================================================================
// Copyright 2019 Autodesk, Inc.  All rights reserved.
// Use of this software is subject to the terms of the Autodesk license agreement
// provided at the time of installation or download, or which otherwise
// accompanies this software in either electronic or hard copy form.
// ==========================================================================
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
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MGlobal.h>
#include <maya/MDagMessage.h>
#include <maya/MEvaluationNode.h>
#include <maya/MEventMessage.h>
#include <maya/MEvaluationManager.h>
#include <maya/MEvaluationNode.h>

// Function sets
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnMessageAttribute.h>

// Viewport 2.0 includes
#include <maya/MDrawRegistry.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MShaderManager.h>
#include <maya/MHWGeometry.h>
#include <maya/MHWGeometryUtilities.h>

// Proxies
#include <maya/MPxTransform.h>
#include <maya/MPxTransformationMatrix.h>
#include <maya/MPxSurfaceShape.h>
#include <maya/MPxDrawOverride.h>

// Lunar
#include "../maya/api/Utils.h"
#include <../maya/api/MText.h>

#include "../maya/api/MAttribute.h"




//-------------------------------------------------------------------------------------------------
//
// Ctrl Transform Node definition
//
//-------------------------------------------------------------------------------------------------

class CtrlNode : public MPxTransform {
    /* Clean transform instance with a custom type_name. */

public:
    // Controls the size of the footprint geometry
    static Attribute    attr_in_line_matrix;
    static Attribute    attr_out_line_matrix;
    // static  MObject  inputSize;  // 'sz'   Input   Distance
    // static  MObject  outputSize; // 'osz'  Output  Distance: outputSize = [](inputSize) -> {return inputSize;}

    // Add your renderer-required-attributes here
    // static MObject inputXX;
    // static MObject outputXX;

    // Utility attribute for viewport
    // [[maya::storable(false)]] [[maya::connectable(false)]] [[maya::hidden]]
    static	MObject      geometryChanging;  // 'gcg' Output	Bool: geometryChanging = [](inputSize) -> {return true;} [*] check notes

    // Attribute dependencies:
    //      inputSize -> outputSize
    //      inputSize -> geometryChanging 
    //
    //      * inputXX -> outputXX
    //      * inputXX -> geometryChanging (if XX affect the geometry)
    //
    // "Logical" dependencies (Technique 1):
    //      outputSize, geometryChanging -> [renderer]
    //      * outputXX -> [renderer]
    //
    // Additional note:
    //
    // Q :  Why is there outputSize ? (Technique 1)
    // A :  Input attributes, like inputSize cannot be cached by Evaluation Cache
    //      Check FootPrintNode::setDependentsDirty() for more details about this work-around
    //
    // Q :  'outputXX' is not updating in EM mode? (Technique 1)
    // A :  The virtual connections to [renderer] are not understand by EM.
    //      Add this connection to FootPrintNode::setDependentsDirty().
    //
    // Q :  "geometryChanging" always returns 'true' ? (Technique 1.1)
    // A :  "geometryChanging" is a "dirty flag attribute" tracking if the node needs geometry-update
    //      If anything affecting it is changed, it will re-evaluate and return 'true' 
    //      This will notify the viewport that the geometry is changing, 
    //      Then, it will be reset to 'false' when geometry is updated (populateGeometry() is called)
    //      This allows us to track the dirty status without override setDependentsDirty() or postEvaluation()
    //      * Note, viewport will not reset its value in background evaluation (VP2 caching)
    //      Check requiresGeometryUpdate(), populateGeometry() for detail


public:
    // Class attributes
    static const MString type_name;
    static const MTypeId type_id;
    static const MString type_drawdb;
    static const MString type_drawid;

    // Node attributes
    // static MObject size;
    static MObject local_position, local_positionX, local_positionY, local_positionZ;
    static MObject local_rotate, local_rotateX, local_rotateY, local_rotateZ;
    static MObject local_scale, local_scaleX, local_scaleY, local_scaleZ;

    static MObject attr_line_width;
    static MObject attr_shape_indx;
    static MObject attr_in_draw_line;

    static MObject attr_draw_solver_mode;
    static MObject attr_solver_mode_size;
    static MObject attr_solver_mode_positionX, attr_solver_mode_positionY, attr_solver_mode_positionZ, attr_solver_mode_position;
    static MObject attrInText;

    static MObject attrInFkIk;
    static MObject attr_component;

    // Use only on dynamic ctrl like fk/ik blending or pole vectors
    bool draw_line;

    MObject self_object;
    MDagPath self_path;

    // Constructors
    CtrlNode()
        : MPxTransform()
    {};
    // Destructors
    virtual ~CtrlNode() override {};

    // Class Methods
    static void *   creator() {return new CtrlNode();};
    static MStatus  initialize();
    virtual void    postConstructor() override;

    MStatus         setDependentsDirty(const MPlug& plugBeingDirtied, MPlugArray& affectedPlugs) override;
    MStatus         compute(const MPlug& plug, MDataBlock& dataBlock) override;
    MStatus         postEvaluation(const  MDGContext& context, const MEvaluationNode& evaluationNode, PostEvaluationType evalType) override; 

    void            getCacheSetup(const MEvaluationNode& evalNode, MNodeCacheDisablingInfo& disablingInfo, MNodeCacheSetupInfo& cacheSetupInfo, MObjectArray& monitoredAttributes) const override;
    SchedulingType  schedulingType() const override {return SchedulingType::kParallel;}

    bool            isBounded() const override {return true;};
    virtual MBoundingBox boundingBox() const override;
};




//-------------------------------------------------------------------------------------------------
//
// Ctrl Draw Override definition
//
//-------------------------------------------------------------------------------------------------

class CtrlUserData : public MUserData {
public:
    MMatrix         mat_local;
    MBoundingBox    bbox;
    MMatrix         mat_pv;
    MPoint          pos_draw_pv_to;

    short           shape_indx;
    unsigned int    prio_depth;
    MPointArray     list_vertecies;
    MPointArray     list_lines;
    float           line_width;
    MColor          col_wireframe;

    // Fk / Ik state
    MObject objDrawLineTo;
    MMatrix matTo;
    double fkIk;
    bool bDrawline;

    bool draw_solver_mode;
    unsigned int solver_mode_size;
    MPoint pos_solver_mode;
    MString str_solver_mode;

    // Constructors
    CtrlUserData()
        : MUserData(false)
    {};

    // Destructor
    virtual ~CtrlUserData() override {};
    virtual void get_plugs(const MObject& object);
    virtual void get_shape(const MObject& object, const MDagPath& dp_object, MMatrix matrix);
    virtual void get_bbox(const MObject& object, const MDagPath& dp_object, MMatrix matrix);
    virtual void get_text(const MObject& object);
};




class CtrlDrawOverride : public MHWRender::MPxDrawOverride {
private:

public:
    // Destructor
    virtual ~CtrlDrawOverride() override {
        ptrCtrlNode = NULL; 
        if (fModelEditorChangedCbId != 0) {
            MMessage::removeCallback(fModelEditorChangedCbId);
            fModelEditorChangedCbId = 0;
        }
    };
    // Public Methods
    static MHWRender::MPxDrawOverride* creator(const MObject& obj) {return new CtrlDrawOverride(obj);}
    virtual MHWRender::DrawAPI supportedDrawAPIs() const override {return MHWRender::kAllDevices;}

    virtual bool isBounded(const MDagPath& objPath, const MDagPath& cameraPath) const override {return true;}
    virtual MBoundingBox boundingBox(
        const MDagPath& objPath,
        const MDagPath& cameraPath
    ) const override;
    virtual bool hasUIDrawables() const override {return true;}
    virtual MUserData* prepareForDraw(
        const MDagPath& objPath,
        const MDagPath& cameraPath,
        const MHWRender::MFrameContext& frameContext,
        MUserData* oldData
    ) override;
    virtual void addUIDrawables(
        const MDagPath& objPath,
        MHWRender::MUIDrawManager& drawManager,
        const MHWRender::MFrameContext& frameContext,
        const MUserData* data
    ) override;

private:
    // Constructors
    CtrlDrawOverride(const MObject& obj)
        : MHWRender::MPxDrawOverride(obj, nullptr, false)
        , ptrCtrlNode(nullptr)
    {
        fModelEditorChangedCbId = MEventMessage::addEventCallback("modelEditorChanged", OnModelEditorChanged, this);
        MStatus status;
        MFnDependencyNode fn_node(obj, &status);
        ptrCtrlNode = status ? dynamic_cast<CtrlNode*>(fn_node.userNode()) : NULL;
    };

    CtrlNode*   ptrCtrlNode;    // The node we are rendering
    MCallbackId fModelEditorChangedCbId;
    static void OnModelEditorChanged(void *clientData);
};
