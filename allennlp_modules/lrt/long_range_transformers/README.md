# APIs

Originally implemented by [idiap](https://github.com/idiap/fast-transformers).

It's not easy to compile. I succeeded with GCC-9.3 with torch 1.10.1 and without CUDA.

Supported transformers:

**Huggingface attention**: type `huggingface`. The HF version, original.

**Full attention**: type `full`. Implemented by fast. Should be equivalent to above, but implemented by fast transformers.

**Linear attention**: type `linear`. Implemented by fast. Use linear kernel instead of exp.
The default feature function is elu, which is not an approximation of full attention.
It is the official implementation of "Linear Transformers".

By changing it to the "FAVOR", we got performer.

Refer to [this doc](https://fast-transformers.github.io/feature_maps/#available-feature-maps) for more functions.

**Reformer**: type `reformer`. Implemented by fast. Type `reformer`.
