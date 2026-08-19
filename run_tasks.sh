#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON_BIN:-python}"
RAG="${RAG:-True}"
EMBED_MODEL="${EMBED_MODEL:-nomic-embed-text}"
LLM_MODEL="${LLM_MODEL:-codellama:7b}"
FRAMEWORK="${FRAMEWORK:-OpenDP}"
RUN_COUNT="${RUN_COUNT:-1}"

safe_path_part() {
  local value
  value="$(printf "%s" "$1" | sed -E 's/[^A-Za-z0-9.-]+/-/g; s/^[.-]+//; s/[.-]+$//')"
  printf "%s" "${value:-unknown}"
}

tasks=(
  "stat_count"
  # "stat_mean"
  # "stat_histogram"
  # "stat_composition"
  # "selection_quantile"
  "selection_topk"
  "ml_regression"
  # "ml_pca"
)

model_part="$(safe_path_part "${LLM_MODEL}")"
rag_part=""
if [[ "${RAG}" == "True" ]]; then
  rag_part="rag"
fi
framework_part="$(safe_path_part "${FRAMEWORK}")"
timestamp="$(date +%s%N)"
output_dir="generated_code/${model_part}_${rag_part}_${framework_part}_${timestamp}"

mkdir -p "${output_dir}"
echo "Saving all generated code to: ${output_dir}"

for task in "${tasks[@]}"; do
  for ((run_index = 1; run_index <= RUN_COUNT; run_index++)); do
    echo "Running task: ${task} (${run_index}/${RUN_COUNT})"
    command=(
      "${PYTHON_BIN}" llamaindex_querying.py
      --rag "${RAG}"
      --llm "${LLM_MODEL}"
      --task "${task}"
      --framework "${FRAMEWORK}"
      --run-index "${run_index}"
      --output-dir "${output_dir}"
      --weave
    )

    if [[ -n "${EMBED_MODEL}" ]]; then
      command+=(--embed "${EMBED_MODEL}")
    fi

    "${command[@]}"
  done
done
