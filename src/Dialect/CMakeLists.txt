add_mlir_dialect_library(MLIRRemoteMem
        RemoteMemDialect.cpp
        FunctionUtils.cpp
        WorkloadAnalysis.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        MLIRRemoteMemIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRMemRefDialect
        MLIRInferTypeOpInterface
)

add_subdirectory(Transforms)