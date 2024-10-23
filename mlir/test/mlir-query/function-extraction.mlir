// RUN: mlir-query %s -c "m hasOpName(\"arith.mulf\").extract(\"testmul\")" | FileCheck %s

// CHECK: func.func @testmul({{.*}}) -> (f32, f32, f32) {
// CHECK:       %[[MUL0:.*]] = arith.mulf {{.*}} : f32
// CHECK:       %[[MUL1:.*]] = arith.mulf {{.*}}, %[[MUL0]] : f32
// CHECK:       %[[MUL2:.*]] = arith.mulf {{.*}} : f32
// CHECK-NEXT:  return %[[MUL0]], %[[MUL1]], %[[MUL2]] : f32, f32, f32


func.func @complexOperation(%x: f32, %y: f32, %z: f32) -> f32 {
  // Arithmetic operations without any multiplication
  %add1 = arith.addf %x, %y : f32
  %sub1 = arith.subf %add1, %z : f32
  %add2 = arith.addf %sub1, %y : f32
  %sub2 = arith.subf %add2, %x : f32
  %add3 = arith.addf %sub2, %z : f32

  // Now %sub3 depends on both %add3 and %sub2, creating more complex def-use chain
  %sub3 = arith.subf %add3, %sub2 : f32

  // Final multiplication operation involving %sub3 and %sub2
  %final_op = arith.mulf %sub3, %sub2 : f32

  return %final_op : f32
}



