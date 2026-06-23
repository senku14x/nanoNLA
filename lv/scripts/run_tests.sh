#!/usr/bin/env bash
# Run the pure-math self-tests (no GPU / no torch required).
# These guard the measurement instruments. Run before trusting any gate number.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
echo "== metrics =="            && python3 -m lv_explainers.metrics
echo "== concepts =="           && python3 -m lv_explainers.concepts
echo "== data =="               && python3 -m lv_explainers.data
echo "== text_baselines =="      && python3 -m lv_explainers.text_baselines
echo "== validate_concepts =="   && python3 -m lv_explainers.validate_concepts
echo "== gate0_counterfactual ==" && python3 -m lv_explainers.gate0_counterfactual
echo "ALL SELF-TESTS PASSED"
