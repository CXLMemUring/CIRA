module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @str0("%d\00") {addr_space = 0 : i32}
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(100 : index) : i64
    %1 = llvm.mlir.constant(0 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(12 : i32) : i32
    %4 = llvm.mlir.undef : i32
    llvm.br ^bb1(%1 : i64)
  ^bb1(%5: i64):  // 2 preds: ^bb0, ^bb4
    %6 = llvm.icmp "slt" %5, %0 : i64
    llvm.cond_br %6, ^bb2, ^bb5
  ^bb2:  // pred: ^bb1
    %7 = llvm.trunc %5 : i64 to i32
    %8 = llvm.icmp "eq" %7, %3 : i32
    llvm.cond_br %8, ^bb3, ^bb4
  ^bb3:  // pred: ^bb2
    %9 = llvm.mlir.addressof @str0 : !llvm.ptr<array<3 x i8>>
    %10 = llvm.getelementptr %9[0, 0] : (!llvm.ptr<array<3 x i8>>) -> !llvm.ptr<i8>
    %11 = llvm.call @printf(%10, %7) : (!llvm.ptr<i8>, i32) -> i32
    llvm.br ^bb4
  ^bb4:  // 2 preds: ^bb2, ^bb3
    %12 = llvm.add %5, %2  : i64
    llvm.br ^bb1(%12 : i64)
  ^bb5:  // pred: ^bb1
    llvm.return %4 : i32
  }
}

