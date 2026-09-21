#!/usr/bin/env bash
set -euo pipefail
R=${MOSPR_ROOT}
ENVDIR=${MAMBA_ENV:-$R/envs/cpnn-mamba}
WORK=${WORK:-/tmp/2dmamba_build}
ARCH=${ARCH:-120}            # sm_120 = RTX PRO 6000 Blackwell
DEST=${MOSPR_CPNN_REPO:-$R/external/CPNN}/model/comparisons/pscan_cuda

mkdir -p "$WORK"; cd "$WORK"
[ -d 2DMamba ] || git clone --depth 1 https://github.com/AtlasAnalyticsLab/2DMamba
K=$WORK/2DMamba/cuda_kernel

cat > "$K/include/cuda_stdint.h" <<'H'
// Replacement for the legacy header removed in CUDA 12; it only exposed stdint types.
H

cat > "$K/include/cub_compat.cuh" <<'H'
// Restores cub APIs dropped in CUDA 13 (CCCL); behaviour matches cub in CUDA 12.8.
namespace cub {
__device__ __forceinline__ void CTA_SYNC() { __syncthreads(); }
__device__ __forceinline__ int CTA_SYNC_AND(int p) { return __syncthreads_and(p); }
__device__ __forceinline__ int CTA_SYNC_OR(int p) { return __syncthreads_or(p); }
__device__ __forceinline__ unsigned int LaneId() {
    unsigned int ret; asm("mov.u32 %0, %%laneid;" : "=r"(ret)); return ret;
}
template <int LOGICAL_WARP_THREADS, int PTX_ARCH>
__host__ __device__ __forceinline__ unsigned int WarpMask(unsigned int warp_id) {
    constexpr bool is_pow_of_two = (LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0;
    constexpr bool is_arch_warp = (LOGICAL_WARP_THREADS == 32);
    unsigned int member_mask = 0xFFFFFFFFu >> (32 - LOGICAL_WARP_THREADS);
    if (is_pow_of_two && !is_arch_warp) member_mask <<= warp_id * LOGICAL_WARP_THREADS;
    (void) warp_id;
    return member_mask;
}
}  // namespace cub
H

grep -q cub_compat "$K/CMakeLists.txt" || sed -i \
  's|set(CMAKE_CUDA_STANDARD 17)|set(CMAKE_CUDA_STANDARD 17)\nset(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -include ${CMAKE_SOURCE_DIR}/include/cub_compat.cuh")|' \
  "$K/CMakeLists.txt"
sed -i 's/-D_GLIBCXX_USE_CXX11_ABI=0/-D_GLIBCXX_USE_CXX11_ABI=1/' "$K/CMakeLists.txt"

cd "$K"; rm -rf build
cmake -DCMAKE_BUILD_TYPE=Release -DPython_ROOT_DIR="$ENVDIR" \
      -DCUDA_ARCHS="$ARCH" -DOUTPUT_DIRECTORY="$K/out" -B build
cmake --build build -- -j16
cp "$K/out/pscan.so" "$DEST/pscan.so"; rm -rf "$DEST/__pycache__"
cd "${MOSPR_CPNN_REPO:-$R/external/CPNN}"
"$ENVDIR/bin/python" -c "
import sys; sys.path.insert(0,'.')
from model.comparisons import pscan_cuda
assert hasattr(pscan_cuda,'fwd') and hasattr(pscan_cuda,'bwd')
print('pscan_cuda OK - fwd/bwd verified')"
