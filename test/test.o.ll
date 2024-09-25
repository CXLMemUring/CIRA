; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@str0 = internal constant [3 x i8] c"%d\00"

declare ptr @malloc(i64)

declare void @free(ptr)

declare i32 @printf(ptr, ...)

define i32 @main() !dbg !3 {
  br label %1, !dbg !7

1:                                                ; preds = %9, %0
  %2 = phi i64 [ %10, %9 ], [ 0, %0 ]
  %3 = icmp slt i64 %2, 100, !dbg !9
  br i1 %3, label %4, label %11, !dbg !10

4:                                                ; preds = %1
  %5 = trunc i64 %2 to i32, !dbg !11
  %6 = icmp eq i32 %5, 12, !dbg !12
  br i1 %6, label %7, label %9, !dbg !13

7:                                                ; preds = %4
  %8 = call i32 (ptr, ...) @printf(ptr @str0, i32 %5), !dbg !14
  br label %9, !dbg !15

9:                                                ; preds = %7, %4
  %10 = add i64 %2, 1, !dbg !16
  br label %1, !dbg !17

11:                                               ; preds = %1
  ret i32 undef, !dbg !18
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "mlir", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "LLVMDialectModule", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "main", linkageName: "main", scope: null, file: !4, line: 4, type: !5, scopeLine: 4, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !6)
!4 = !DIFile(filename: "test.o_llvm.mlir", directory: "/home/yangyw/isca25/CIRA/test")
!5 = !DISubroutineType(types: !6)
!6 = !{}
!7 = !DILocation(line: 10, column: 5, scope: !8)
!8 = !DILexicalBlockFile(scope: !3, file: !4, discriminator: 0)
!9 = !DILocation(line: 12, column: 10, scope: !8)
!10 = !DILocation(line: 13, column: 5, scope: !8)
!11 = !DILocation(line: 15, column: 10, scope: !8)
!12 = !DILocation(line: 16, column: 10, scope: !8)
!13 = !DILocation(line: 17, column: 5, scope: !8)
!14 = !DILocation(line: 21, column: 11, scope: !8)
!15 = !DILocation(line: 22, column: 5, scope: !8)
!16 = !DILocation(line: 24, column: 11, scope: !8)
!17 = !DILocation(line: 25, column: 5, scope: !8)
!18 = !DILocation(line: 27, column: 5, scope: !8)
