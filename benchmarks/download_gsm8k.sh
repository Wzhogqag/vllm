#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "$SCRIPT_DIR/.." && pwd)
DATASET_DIR=${DATASET_DIR:-"$ROOT_DIR/log/datasets/gsm8k"}

TRAIN_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
TEST_URL="https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

mkdir -p "$DATASET_DIR"

curl --fail --location --connect-timeout 20 --max-time 300 \
    "$TRAIN_URL" \
    -o "$DATASET_DIR/train.jsonl"

curl --fail --location --connect-timeout 20 --max-time 300 \
    "$TEST_URL" \
    -o "$DATASET_DIR/test.jsonl"

wc -l "$DATASET_DIR/train.jsonl" "$DATASET_DIR/test.jsonl"