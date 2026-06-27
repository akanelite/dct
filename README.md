# Discrete Cosine Transform

This library implements the [Discrete Cosine Transform](https://www.cse.iitd.ac.in/~pkalra/col783-2017/DCT-Paper.pdf) for PyTorch, 
leveraging its highly optimized convolution operations to perform fast 2D DCT on both CPU and GPU.

While 2D DCT is typically implemented as two successive 1D DCTs, it is hard to achieve high performance without low-level optimizations or hardware acceleration. 
This implementation builds convolutional filters by unrolling a 2D basis derived from the Kronecker product of 1D bases. 
It takes advantage of the highly optimized convolution backends available in deep learning frameworks.

By default the transform is orthonormal (`norm="ortho"`, energy-preserving); pass `norm="backward"` or `"forward"` for the corresponding SciPy conventions.
