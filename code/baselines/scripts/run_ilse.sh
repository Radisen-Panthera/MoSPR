#!/usr/bin/env bash
set -uo pipefail
cd "${MOSPR_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export FILTERED=1 ILSE=1 ONLY="ilse_mean ilse_max" MAXJOBS=${MAXJOBS:-12}
CONDITION=psplit    bash code/baselines/scripts/axis_baselines.sh
CONDITION=psplit_tv bash code/baselines/scripts/axis_baselines.sh
