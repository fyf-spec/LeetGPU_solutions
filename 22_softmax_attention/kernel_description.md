# Softmax Attention

Medium

Implement a GPU program that computes the softmax attention operation for a given set of matrices. Given the query matrix `Q` of size `M×d`, key matrix `K` of size `N×d`, and value matrix `V` of size `N×d`, your program should compute the output matrix using the formula:where the softmax function is applied row-wise.

## Implementation Requirements

- Use only GPU native features (external libraries are not permitted)
- The `solve` function signature must remain unchanged
- The final result must be stored in the output matrix `output`

## Example 1:

**Input:**
`Q` (2×4):`K` (3×4):`V` (3×4):

**Output:**
`output` (2×4):

## Example 2:

**Input:**
`Q` (1×2):`K` (2×2):`V` (2×2):

**Output:**
`output` (1×2):

## Constraints

- Matrix `Q` is of size `M×d` and matrices `K` and `V` are of size `N×d`
- 1 ≤ `M`, `N` ≤ 100,000
- 1 ≤ `d` ≤ 128
- Performance is measured with `M` = 512, `N` = 256