module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("%d\00") {addr_space = 0 : i32}
  func.func @main() -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c100 = arith.constant 100 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c12_i32 = arith.constant 12 : i32
    %0 = llvm.mlir.undef : i32
    scf.for %arg0 = %c0 to %c100 step %c1 {
      %1 = arith.index_cast %arg0 : index to i32
      %2 = arith.cmpi eq, %1, %c12_i32 : i32
      scf.if %2 {
        %3 = llvm.mlir.addressof @str0 : !llvm.ptr<array<3 x i8>>
        %4 = llvm.getelementptr %3[0, 0] : (!llvm.ptr<array<3 x i8>>) -> !llvm.ptr<i8>
        %5 = llvm.call @printf(%4, %1) : (!llvm.ptr<i8>, i32) -> i32
      }
    }
    return %0 : i32
  }
}
