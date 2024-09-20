add_mlir_dialect_library(MLIRRMEMTransforms
        SearchRemote.cpp
        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include

        DEPENDS
        RemoteMemPassIncGen

        LINK_LIBS PUBLIC
        MLIRAnalysis
        MLIRDataLayoutInterfaces
        MLIRMemRefDialect
        MLIRLLVMDialect
        MLIRVectorDialect
        MLIRAffineDialect
        MLIRSCFDialect
        MLIRRemoteMem
        MLIRTransforms
)