#!/usr/bin/env bash
set -uo pipefail
ENV=${MAMBA_ENV:-./envs/cpnn-mamba}
SRC=${SRC:-/tmp/mambamil_fork}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.0;9.0;12.0"
export MAX_JOBS="${MAX_JOBS:-48}"

[ -d "$SRC" ] || git clone --depth 1 https://github.com/isyangshu/MambaMIL.git "$SRC"

python3 - "$SRC/mamba/setup.py" <<'PY'
import pathlib,sys,re
p=pathlib.Path(sys.argv[1]); t=p.read_text()
if "compute_120" in t: print("  [1] already patched"); raise SystemExit
for a in ("53","62","70","72","87"):
    t=t.replace(f'    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_{a},code=sm_{a}")\n','')
t=t.replace('        cc_flag.append("arch=compute_90,code=sm_90")',
 '        cc_flag.append("arch=compute_90,code=sm_90")\n'
 '    if bare_metal_version >= Version("12.8"):\n'
 '        cc_flag.append("-gencode")\n'
 '        cc_flag.append("arch=compute_120,code=sm_120")')
p.write_text(t); print("  ① setup.py → sm_80/90/120")
PY

sed -i 's/cub::CTA_SYNC()/__syncthreads()/g; s/cub::LaneId()/(threadIdx.x \& 0x1f)/g' \
    "$SRC/mamba/csrc/selective_scan/reverse_scan.cuh"
echo "  [2] reverse_scan.cuh -> CUB compatibility"

echo "  -> building causal-conv1d 1.1.1 (version the fork requires)"
"$ENV/bin/pip" install --no-build-isolation --no-deps causal-conv1d==1.1.1 2>&1 | tail -2
echo "  -> building the mamba_ssm fork"
( cd "$SRC/mamba" && "$ENV/bin/pip" install --no-build-isolation --no-deps . ) 2>&1 | tail -2

SP="$ENV/lib/python3.11/site-packages/mamba_ssm"

grep -rl 'from mamba\.mamba_ssm' "$SP" 2>/dev/null \
  | xargs -r sed -i 's/from mamba\.mamba_ssm/from mamba_ssm/g'
echo "  [3] fixing import paths"

python3 - "$SP/__init__.py" <<'PY'
import pathlib,sys
p=pathlib.Path(sys.argv[1]); t=p.read_text()
old="from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel"
if old in t:
    t=t.replace(old,"try:\n    "+old+"\nexcept ImportError:\n    MambaLMHeadModel = None")
    p.write_text(t); print("  [4] LM head import made optional")
else: print("  [4] already patched")
PY

echo; echo "=== verification ==="
"$ENV/bin/python" -c "
import torch
from mamba_ssm import SRMamba, BiMamba
x = torch.randn(2, 512, 64, device='cuda')
assert SRMamba(d_model=64).cuda()(x).shape == x.shape
assert BiMamba(d_model=64).cuda()(x).shape == x.shape
print('  OK: SRMamba / BiMamba run on Blackwell')
" 2>&1 | grep -aE 'OK:|Error' | tail -3
