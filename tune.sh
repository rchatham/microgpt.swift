#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-model-32-10000.json}
BASE=(N_EMBD=32 N_HEAD=4 NUM_STEPS=0 LOAD_CHECKPOINT="$MODEL" MIN_LENGTH=4 MAX_LENGTH=8 MAX_REPEAT=2 UNIQUE=true REQUIRE_VOWEL=true SAMPLES=20)

run() {
  echo
  echo "=== $* ==="
  env "${BASE[@]}" "$@" .build/release/microgpt.swift | sed -n '/--- inference/,$p'
}

run SEED=1 TEMPERATURE=0.55 TOP_K=8 TOP_P=0.9
run SEED=2 TEMPERATURE=0.65 TOP_K=8 TOP_P=0.9
run SEED=3 TEMPERATURE=0.70 TOP_K=10 TOP_P=0.9
run SEED=4 TEMPERATURE=0.75 TOP_K=12 TOP_P=0.95
run SEED=5 PREFIX=ka TEMPERATURE=0.65 TOP_K=8 TOP_P=0.9
run SEED=6 PREFIX=sha TEMPERATURE=0.70 TOP_K=10 TOP_P=0.9
