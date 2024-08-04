#ifndef RMEM_DIALECT_TRANSFORM_PASSES
#define RMEM_DIALECT_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def RMEMDeterminePoolSource : Pass<"rmem-decide-pool", "mlir::ModuleOp"> {
  let summary = "Choose a cache pool for each remote allocation";
  let constructor = "mlir::createRemoteMemDPSPass()";
  let dependentDialects = ["LLVM::LLVMDialect", "rmem::RemoteMemDialect", "func::FuncDialect"];
}

#endif