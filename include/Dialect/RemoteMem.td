#ifndef REMOTEMEMREF_BASE
#define REMOTEMEMREF_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/DialectBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/LLVMIR/LLVMInterfaces.td"

def RemoteMem_Dialect : Dialect {
  let name = "cira";
  let cppNamespace = "::mlir::cira";
  let description = [{The `cira` dialect provides a unified interface to the remote memory access. The dialect will be converted to raw pointer or remotable access eventually}];
  let extraClassDeclaration = [{
  private:
    void registerTypes();

  public:
  }];
  // let emitAccessorPrefix = kEmitAccessorPrefix_Prefixed;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}
// basic block top to remote memory

/* RemoteMemRef Type Definitions */
def IsRemoteMemRefTypePred : CPred<"$_self.isa<::mlir::cira::RemoteMemRefType>()">;

class RemoteMemRefOf<list<Type> allowedTypes> :
  ContainerType<
    AnyTypeOf<allowedTypes>,
    IsRemoteMemRefTypePred,
    "$_self.cast<::mlir::cira::RemoteMemRefType>().getElementType()",
    "Remote mem ref of accepted type",
    "::mlir::cira::RemoteMemRefType">;

// For future extensions
def AnyRemoteMemRef : RemoteMemRefOf<[AnyType]>;

// Used for now
// AllowedRemoteElement = [AnyMemRef, LLVM_AnyPointer]
def RemoteMemRef : RemoteMemRefOf<[AnyMemRef, LLVM_AnyPointer]>;

class Detail_RemoteMemType<string name, string typeMnemonic, list<Trait> traits = []>
  :TypeDef<RemoteMem_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Detail_RemoteMemRefType: Detail_RemoteMemType<"RemoteMemRef", "rmref", [
  /* Add following interfaces so that remote memref can be the member of llvm.ptr and memref */
  DeclareTypeInterfaceMethods<LLVM_PointerElementTypeInterface>,
  DeclareTypeInterfaceMethods<MemRefElementTypeInterface>
]> {
  let summary = "Remote MemRef Type";
  let parameters = (ins "Type":$elementType,
    "unsigned":$canRemote);
  let genVerifyDecl = 1;
  let assemblyFormat = "`<` $canRemote`,` $elementType `>`";
  // let hasCustomAssemblyFormat = 1;
  let builders = [
    TypeBuilderWithInferredContext<(ins "Type":$elementType,
      CArg<"unsigned", "1">:$canRemote), [{
      return Base::get(elementType.getContext(), elementType, canRemote);
    }]>
  ];
  let extraClassDeclaration = [{
    static bool isValidElementType(Type elementType);
    Type getRawSourceType();
    Type getInnerElementType();
  }];

}


#endif
