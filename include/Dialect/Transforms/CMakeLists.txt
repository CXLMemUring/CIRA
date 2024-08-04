set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name RemoteMem)
add_public_tablegen_target(RemoteMemPassIncGen)
add_dependencies(mlir-headers RemoteMemPassIncGen)
