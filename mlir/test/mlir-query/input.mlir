// RUN: mlir-query %s -c "m anyOf(getAllDefinitions(hasOpName(\"arith.addf\"),2),getAllDefinitions(hasOpName(\"tensor.extract\"),1))" | FileCheck %s

func.func @slicing_linalg_op(%arg0 : index, %arg1 : index, %arg2 : index) {
  %a = memref.alloc(%arg0, %arg2) : memref<?x?xf32>
  %b = memref.alloc(%arg2, %arg1) : memref<?x?xf32>
  %c = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  %d = memref.alloc(%arg0, %arg1) : memref<?x?xf32>
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%c : memref<?x?xf32>)
  linalg.matmul ins(%a, %b : memref<?x?xf32>, memref<?x?xf32>)
               outs(%d : memref<?x?xf32>)
  memref.dealloc %c : memref<?x?xf32>
  memref.dealloc %b : memref<?x?xf32>
  memref.dealloc %a : memref<?x?xf32>
  memref.dealloc %d : memref<?x?xf32>
  return
}
// CHECK: Match #1:
// CHECK: %[[LINALG:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} 
// CHECK-SAME: ins(%arg0 : tensor<5x5xf32>) outs(%arg1 : tensor<5x5xf32>)

// CHECK: {{.*}}.mlir:7:10: note: "root" binds here
// CHECK: %[[ADDF1:.*]] = arith.addf %in, %in : f32

// CHECK: Match #2:
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[LINALG]] {{\[\[.*\]\]}} : tensor<5x5xf32> into tensor<25xf32>
// CHECK: %[[C2:.*]] = arith.constant {{.*}} : index

// CHECK: {{.*}}.mlir:14:18: note: "root" binds here
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[COLLAPSED]][%[[C2]]] : tensor<25xf32>

// CHECK: Match #3:
// CHECK: %[[COLLAPSED:.*]] = tensor.collapse_shape %[[LINALG]] {{\[\[.*\]\]}} : tensor<5x5xf32> into tensor<25xf32>
// CHECK: %[[C2:.*]] = arith.constant {{.*}} : index
// CHECK: %[[EXTRACTED:.*]] = tensor.extract %[[COLLAPSED]][%[[C2]]] : tensor<25xf32>

// CHECK: {{.*}}.mlir:15:10: note: "root" binds here
// CHECK: %[[ADDF2:.*]] = arith.addf %[[EXTRACTED]], %[[EXTRACTED]] : f32  
