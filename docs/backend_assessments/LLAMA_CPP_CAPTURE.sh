#!/usr/bin/env bash
set -euo pipefail

# OMNIAMIND — llama.cpp local capture
# Author: Massimiliano Brighindi
# Project: MB-X.01

HOST="${HOST:-http://127.0.0.1:8080}"
OUT_DIR="${OUT_DIR:-data}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/llama_cpp_raw_capture_001.json}"

mkdir -p "$OUT_DIR"

PROMPT="${PROMPT:-Return only one word: EVEN or ODD. Question: A box contains 7 red balls and 8 blue balls. Two balls are removed without replacement. Is the probability that both removed balls are blue greater than 1/4?}"

echo "[1/3] Checking llama-server availability at $HOST ..."
curl -sSf "$HOST/health" >/dev/null 2>&1 || {
  echo "ERROR: llama-server is not reachable at $HOST"
  echo "Start it first, e.g.: ./llama-server -m /path/to/model.gguf --port 8080"
  exit 1
}

echo "[2/3] Sending local capture request ..."
curl -sSf "$HOST/completion" \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": $(printf '%s' "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),
    \"n_predict\": 3,
    \"temperature\": 0,
    \"seed\": 42,
    \"n_probs\": 20,
    \"stream\": false
  }" > "$OUT_FILE"

echo "[3/3] Capture saved to $OUT_FILE"

echo
echo "Suggested quick checks:"
echo "  jq '.' $OUT_FILE"
echo "  jq '.completion_probabilities' $OUT_FILE"
echo "  jq '.completion_probabilities | length' $OUT_FILE"
echo
echo "Done."
