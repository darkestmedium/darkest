# Define the project name
project(darkmayaapi LANGUAGES CXX)

set(darkmayaapi_files
	# Api
	MAttribute.h
	MGlobal.h
	MAnimControl.h
	MObject.h
	MPlugin.h
	MRigUtils.h
	MScene.h
	MSolve.h
	MText.h
	MathUtility.h
	Utils.h
	# gl3.h
)

# # Add the source files for the nested project
# add_library(lunarmayaapi SHARED ${API_FILES})

# # Link any dependencies needed for the nested project
# target_link_libraries(lunarmayaapi PRIVATE Maya::Maya)