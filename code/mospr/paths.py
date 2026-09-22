import os
from pathlib import Path

CODE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("MOSPR_ROOT", CODE.parent.parent))
DATA = Path(os.environ.get("MOSPR_DATASET_ROOT", ROOT / "data"))
RESULTS = Path(os.environ.get("MOSPR_RESULTS_ROOT", ROOT / "results/run"))
TABLES = Path(os.environ.get("MOSPR_TABLES_ROOT", ROOT / "results"))
GENESETS = Path(os.environ.get("MOSPR_GENESETS", ROOT / "refdata/genesets"))
CPNN_REPO = Path(os.environ.get("MOSPR_CPNN_REPO", ROOT / "external/CPNN"))
LOGS = Path(os.environ.get("MOSPR_LOGS", RESULTS / "logs"))
PROCESSED = Path(os.environ.get("MOSPR_PROCESSED", ROOT / "resources/processed"))
