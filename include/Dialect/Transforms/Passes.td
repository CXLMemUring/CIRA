#ifndef TRANSFORM_PASSES
#define TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RMEMSearchRemote : Pass<"search-rmem", "ModuleOp"> {
  let summary = "Mark operation as remote target if hit remote type";
  let constructor = "mlir::createRemoteMemSearchRemotePass()";
  let dependentDialects = ["LLVM::LLVMDialect", "cira::RemoteMemDialect"];
}

#endif
