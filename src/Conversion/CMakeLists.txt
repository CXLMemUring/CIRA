add_mlir_conversion_library(MLIRMemDisagg
  CIRA.cpp


  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  DEPENDS
  RemoteMemConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRDataLayoutInterfaces
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransforms
)