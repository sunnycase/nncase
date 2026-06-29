# PyNTT

PyNTT is the Python runtime helper package for generated Triton models from the
nncase `pyntt` target.

M3 provides the importable package, generated-model construction path, static
spec/runtime validation, output allocation, and Triton execution for simple
same-shape contiguous unary, binary, cast, and where kernels. The generated
model owns the generated top kernels and launch metadata directly.
