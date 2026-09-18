set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ARCHIVE=${1:-$ROOT/wrds.zip}
for MARKET in US_SP500 KR_KOSPI_200 JP_Nikkei_225 DE_DAX; do
  bash "$ROOT/scripts/$MARKET.sh" "$ARCHIVE"
done
