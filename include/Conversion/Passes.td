#define REMOTE_MEM_PASSES

include "mlir/Pass/PassBase.td"

def CIRA : Pass<"convert-target-to-remote", "ModuleOp"> {
  let summary = "Will calculate the complexity and mark remote";
  let description = "converts target to cira dialect";
  let constructor = "mlir::cira::createCIRAPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect", 
    "cira::RemoteMemDialect", 
    "func::FuncDialect", 
    "memref::MemRefDialect", 
    "scf::SCFDialect"
  ];
}