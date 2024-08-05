#ifndef RMEM_DIALECT_TRANSFORM_PASSES
#define RMEM_DIALECT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RMEMSearchRemote : Pass<"search-rmem", "ModuleOp"> {
  let summary = "Mark operation as remote target if hit remote type";
  let constructor = "mlir::createRemoteMemSearchRemotePass()";
  let dependentDialects = ["LLVM::LLVMDialect", "rmem::RemoteMemDialect"];
}

#endif
