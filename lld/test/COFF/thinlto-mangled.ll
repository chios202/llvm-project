; REQUIRES: x86
; RUN: mkdir -p %t.dir
; RUN: opt -thinlto-bc %s -o %t.obj
; RUN: opt -thinlto-bc %S/Inputs/thinlto-mangled-qux.ll -o %t.dir/thinlto-mangled-qux.obj
; RUN: lld-link -out:%t.exe -entry:main %t.obj %t.dir/thinlto-mangled-qux.obj

target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc19.0.24215"

%"class.bar" = type { ptr, ptr, ptr, ptr, i32 }

define i32 @main() {
  ret i32 0
}

define available_externally zeroext i1 @"\01?x@bar@@UEBA_NXZ"(ptr %this) unnamed_addr align 2 {
  ret i1 false
}
