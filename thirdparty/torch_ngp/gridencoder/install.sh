export CFLAGS="-O3"
export CXXFLAGS="-O3 -std=c++17"
export CUDAFLAGS="--std=c++17"
export TORCH_CUDA_ARCH_LIST="7.0"
python -m pip install -v --no-build-isolation -e .
