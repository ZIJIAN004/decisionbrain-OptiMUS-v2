#!/usr/bin/env bash
# Launch the FrontierOR adapter detached, with everything one run produces kept
# together in one directory outside the repo: the report, the per-task OptiMUS
# logs, the console output, and a record of what was run.
#
#   ./adapters/frontieror/run.sh --only delage2022
#   JOBS=4 ./adapters/frontieror/run.sh
#
# The converted inputs are NOT run output -- they stay in data/frontieror/ under
# version control, because they are the frozen artifact every evaluated model
# reads.
set -euo pipefail

REPO="${REPO:-/home/bhz/baselines/OptiMUS-v2}"
RUNS_ROOT="${RUNS_ROOT:-/home/bhz/baselines/optimus-runs}"
ENV_FILE="${ENV_FILE:-/home/bhz/Decision Brain/.env}"
PYTHON="${PYTHON:-/home/bhz/miniforge3/envs/decisionbrain_baseline/bin/python}"
JOBS="${JOBS:-1}"

case "$JOBS" in
    ''|*[!0-9]*) echo "JOBS must be an integer from 1 to 24" >&2; exit 1 ;;
esac
if [ "$JOBS" -lt 1 ] || [ "$JOBS" -gt 24 ]; then
    echo "JOBS must be an integer from 1 to 24" >&2
    exit 1
fi

for path in "$REPO" "$ENV_FILE" "$PYTHON"; do
    [ -e "$path" ] || { echo "missing: $path" >&2; exit 1; }
done

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

cd "$REPO"
set -a; . "$ENV_FILE"; set +a
export ADAPTER_JOBS="$JOBS"
CGROUP_SLICE="optimus-${RUN_ID}.slice"
export ADAPTER_CGROUP_SLICE="$CGROUP_SLICE"
systemctl --user set-property --runtime "$CGROUP_SLICE" \
    MemoryMax=100G MemorySwapMax=0

# What produced these numbers, written next to them. A report without the commit
# and the model that made it cannot be compared against anything later.
{
    echo "commit     $(git rev-parse --short HEAD)"
    echo "started    $(date -Is)"
    echo "host       $(hostname)"
    echo "python     $PYTHON"
    echo "jobs       $JOBS"
    echo "memory_gb  100 shared parent cgroup"
    echo "cgroup     $CGROUP_SLICE"
    echo "solver_llm ${LLM_CHAT_MODEL:-unset}"
    echo "converter  ${ADAPTER_CONVERTER_MODEL:-deepseek-v4-flash}"
    echo "args       $*"
} > "$RUN_DIR/run.info"

# -u so console.log is readable while the run is still going.
nohup systemd-run --user --scope -q \
    --slice "${CGROUP_SLICE%.slice}" \
    "$PYTHON" -u -m adapters.frontieror.schedule \
    --jobs "$JOBS" \
    --run-dir "$RUN_DIR" \
    "$@" \
    > "$RUN_DIR/console.log" 2>&1 &

echo $! > "$RUN_DIR/pid"
ln -sfn "$RUN_DIR" "$RUNS_ROOT/latest"

cat "$RUN_DIR/run.info"
echo
echo "run dir : $RUN_DIR   (also $RUNS_ROOT/latest)"
echo "pid     : $(cat "$RUN_DIR/pid")"
echo
echo "  tail -f $RUNS_ROOT/latest/console.log"
echo "  cat     $RUNS_ROOT/latest/report.jsonl"
