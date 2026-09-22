#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=${PY:-python}
export MOSPR_RESULTS_ROOT=results/tables/final/alpha0/source
$PY code/tables/make_tables.py
$PY code/tables/table1_variants.py
$PY code/tables/stats_tables.py
$PY code/tables/refit_tables.py
$PY code/tables/refit_pathway_tables.py
$PY code/tables/robustness_tables.py --collection hallmark
$PY code/tables/robustness_tables.py --collection kegg
for c in BRCA KIRC LUAD; do
  MOSPR_RESULTS_ROOT=results/tables/data_efficiency $PY code/figures/data_efficiency.py --cohort "$c"
done
echo "tables: results/tables/  figures: results/figures/"
