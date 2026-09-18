set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCHIVE=${1:-$ROOT/wrds.zip}
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -u "$ROOT/run.py" --task_name all --market JP_Nikkei_225 --archive_path "$ARCHIVE"
