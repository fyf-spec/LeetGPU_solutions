# INT8 Quantized MatMul

Medium

Implement a quantized matrix multiplication program for 8-bit signed integer matrices. Given two input matrices `A` of dimensions and `B` of dimensions , quantization scales `scale_A`, `scale_B`, output scale `scale_C`, zero-points `zero_point_A`, `zero_point_B`, `zero_point_C`, compute:where `s_A = scale_A`, `z_A = zero_point_A`, etc.

## Implementation Requirements

- External libraries are not permitted
- The `solve` function signature must remain unchanged
- The final result must be stored in the output matrix `C` as `int8`
- After accumulation in int32 and scaling in float32, values must be rounded to the nearest integer, shifted by `zero_point_C`, and clamped to the `[-128, 127]` range

$$
C_{quant} = clamp(round(\dfrac{\sum_{k=0}^{K-1} (A_{ik}-z_A)(B_{kj} - z_B)}{s_C}))
$$



## Example 1:

```
     Input:
     A = [[1, 2],
          [3, 4]]
     B = [[5, 6],
          [7, 8]]
     M = 2, N = 2, K = 2
     scale_A = 0.1, scale_B = 0.2, scale_C = 0.05
     zero_point_A = 0, zero_point_B = 0, zero_point_C = 0

     Output:
     C = [[19, 22],
          [43, 50]]
     
```

## Example 2:

```
     Input:
     A = [[1, 2]]
     B = [[3],
          [4]]
     M = 1, N = 1, K = 2
     scale_A = 1.0, scale_B = 1.0, scale_C = 1.0
     zero_point_A = 1, zero_point_B = 3, zero_point_C = 5

     Output:
     C = [[6]]
     
```

## Constraints

- 1 ≤ `M`, `N`, `K` ≤ 4096
- `scale_A`, `scale_B`, `scale_C` are positive floats
- `-128` ≤ `zero_point_A`, `zero_point_B`, `zero_point_C` ≤ `127`
- Performance is measured with `K` = 2,048, `M` = 8,192, `N` = 4,096