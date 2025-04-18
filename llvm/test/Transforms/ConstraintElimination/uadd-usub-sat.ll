; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 5
; RUN: opt -passes=constraint-elimination -S %s | FileCheck %s

declare i64 @llvm.uadd.sat.i64(i64, i64)
declare i64 @llvm.usub.sat.i64(i64, i64)

define i1 @uadd_sat_uge(i64 %a, i64 %b) {
; CHECK-LABEL: define i1 @uadd_sat_uge(
; CHECK-SAME: i64 [[A:%.*]], i64 [[B:%.*]]) {
; CHECK-NEXT:    [[ADD_SAT:%.*]] = call i64 @llvm.uadd.sat.i64(i64 [[A]], i64 [[B]])
; CHECK-NEXT:    [[CMP:%.*]] = and i1 true, true
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %add.sat = call i64 @llvm.uadd.sat.i64(i64 %a, i64 %b)
  %cmp1 = icmp uge i64 %add.sat, %a
  %cmp2 = icmp uge i64 %add.sat, %b
  %cmp = and i1 %cmp1, %cmp2
  ret i1 %cmp
}

define i1 @usub_sat_ule_lhs(i64 %a, i64 %b) {
; CHECK-LABEL: define i1 @usub_sat_ule_lhs(
; CHECK-SAME: i64 [[A:%.*]], i64 [[B:%.*]]) {
; CHECK-NEXT:    [[SUB_SAT:%.*]] = call i64 @llvm.usub.sat.i64(i64 [[A]], i64 [[B]])
; CHECK-NEXT:    ret i1 true
;
  %sub.sat = call i64 @llvm.usub.sat.i64(i64 %a, i64 %b)
  %cmp = icmp ule i64 %sub.sat, %a
  ret i1 %cmp
}

; Negative test
define i1 @usub_sat_not_ule_rhs(i64 %a, i64 %b) {
; CHECK-LABEL: define i1 @usub_sat_not_ule_rhs(
; CHECK-SAME: i64 [[A:%.*]], i64 [[B:%.*]]) {
; CHECK-NEXT:    [[SUB_SAT:%.*]] = call i64 @llvm.usub.sat.i64(i64 [[A]], i64 [[B]])
; CHECK-NEXT:    [[CMP:%.*]] = icmp ule i64 [[SUB_SAT]], [[B]]
; CHECK-NEXT:    ret i1 [[CMP]]
;
  %sub.sat = call i64 @llvm.usub.sat.i64(i64 %a, i64 %b)
  %cmp = icmp ule i64 %sub.sat, %b
  ret i1 %cmp
}
