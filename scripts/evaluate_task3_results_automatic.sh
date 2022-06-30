#!/bin/bash
# A helper script for the evaluation of Task 3 results
set -e -o xtrace -o pipefail

mkdir -p task3-automatic-results

evaluate() {
  local BASENAME="$1"
  local EXCLUDED_TASK1_RUNS="$2"
  local EXCLUDED_TASK3_RUNS="$3"

  if [[ -e "task3-automatic-results/$BASENAME.relevant_answers" ]]
  then
    return
  fi

  trap 'rm "task3-automatic-results/$BASENAME.relevant_answers"' EXIT

  python3 scripts/evaluate_task3_results_automatic.py \
    -all_task1_answers "runs/task1_arqmath3/" \
    -all_task3_answers "runs/task3_arqmath3/" \
    -collection "collection/" \
    -in "runs/task3_arqmath3/$BASENAME.tsv" \
    -excluded_task1_run_ids "$EXCLUDED_TASK1_RUNS" \
    -excluded_task3_run_ids "$EXCLUDED_TASK3_RUNS" \
    -excluded_topics '["A.302", "A.303", "A.305", "A.308", "A.309", "A.310", "A.314", "A.316", "A.317", "A.318", "A.324", "A.325", "A.327", "A.332", "A.333", "A.338", "A.342", "A.345", "A.346", "A.348", "A.349", "A.352", "A.354", "A.357", "A.358", "A.361", "A.362", "A.363", "A.366", "A.368", "A.369", "A.370", "A.376", "A.379", "A.381", "A.382", "A.383", "A.384", "A.385", "A.388", "A.391", "A.394", "A.399"]' \
    -use_task1_answers false \
    -map "data/teams_document_id.tsv" \
    -task1_qrel "data/qrel_task1_2022_official.tsv" \
    -task3_qrel "data/qrel_task3_2022_official_complete.tsv" \
    -relevant_answer_dump "task3-automatic-results/$BASENAME.relevant_answers" \
    |& tee "task3-automatic-results/$BASENAME.output"

  trap '' EXIT
}

# Baselines
evaluate Baseline2022-task3-GPT3-auto-both-generate-P '[]' '["GPT3"]'

# Approach0
APPROACH0_EXCLUDED_TASK1_RUNS='[
  "APPROACH0",
  "maprun_arqmath3_to_colbert",
  "search_arqmath3_colbert-APPROACH0"
]'
APPROACH0_EXCLUDED_TASK3_RUNS='[
  "maprun_arqmath3_to_colbert__select_sentence",
  "maprun_arqmath3_to_colbert__select_sentence_from_beginning",
  "maprun_arqmath3_to_colbert_top1",
  "search_arqmath3_colbert_top1"
]'

evaluate approach0-task3-run1-manual-both-extract-A \
  "$APPROACH0_EXCLUDED_TASK1_RUNS" \
  "$APPROACH0_EXCLUDED_TASK3_RUNS"
evaluate approach0-task3-run2-manual-both-extract-A \
  "$APPROACH0_EXCLUDED_TASK1_RUNS" \
  "$APPROACH0_EXCLUDED_TASK3_RUNS"
evaluate approach0-task3-run3-manual-both-extract-A \
  "$APPROACH0_EXCLUDED_TASK1_RUNS" \
  "$APPROACH0_EXCLUDED_TASK3_RUNS"
evaluate approach0-task3-run4-manual-both-extract-A \
  "$APPROACH0_EXCLUDED_TASK1_RUNS" \
  "$APPROACH0_EXCLUDED_TASK3_RUNS"
evaluate approach0-task3-run5-manual-both-extract-P \
  "$APPROACH0_EXCLUDED_TASK1_RUNS" \
  "$APPROACH0_EXCLUDED_TASK3_RUNS"

# DPRL
DPRL_EXCLUDED_TASK1_RUNS='[
  "AMRQASIM",
  "QASim",
  "QQAMR",
  "RRAMRSVM",
  "SVM"
]'
DPRL_EXCLUDED_TASK3_RUNS='[
  "AMRBERT",
  "AMRSBERT",
  "SVMBERT",
  "SVMSBERT"
]'

evaluate DPRL-Task3-AMRBERT-auto-both-extract-A \
  "$DPRL_EXCLUDED_TASK1_RUNS" \
  "$DPRL_EXCLUDED_TASK3_RUNS"
evaluate DPRL-Task3-AMRSBERT-auto-both-extract-A \
  "$DPRL_EXCLUDED_TASK1_RUNS" \
  "$DPRL_EXCLUDED_TASK3_RUNS"
evaluate DPRL-Task3-SVMBERT-auto-both-extract-P \
  "$DPRL_EXCLUDED_TASK1_RUNS" \
  "$DPRL_EXCLUDED_TASK3_RUNS"
evaluate DPRL-Task3-SVMSBERT-auto-both-extract-A \
  "$DPRL_EXCLUDED_TASK1_RUNS" \
  "$DPRL_EXCLUDED_TASK3_RUNS"

# TU_DBS
TU_DBS_EXCLUDED_TASK1_RUNS='[
  "base_10",
  "Khan_SE_10",
  "math_10",
  "math_10_add",
  "roberta_10"
]'
TU_DBS_EXCLUDED_TASK3_RUNS='[
  "amps3_se1_hints",
  "amps3_se1_len_pen_20_sample_hint",
  "se3_len_pen_10",
  "shortest"
]'

evaluate TU_DBS-task3-amps3_se1_hints-auto-both-generate-A \
  "$TU_DBS_EXCLUDED_TASK1_RUNS" \
  "$TU_DBS_EXCLUDED_TASK3_RUNS"
evaluate TU_DBS-task3-amps3_se1_len_pen_20_sample_hint-auto-both-generate-A \
  "$TU_DBS_EXCLUDED_TASK1_RUNS" \
  "$TU_DBS_EXCLUDED_TASK3_RUNS"
evaluate TU_DBS-task3-se3_len_pen_10-auto-both-generate-A \
  "$TU_DBS_EXCLUDED_TASK1_RUNS" \
  "$TU_DBS_EXCLUDED_TASK3_RUNS"
evaluate TU_DBS-task3-shortest-auto-both-generate-P \
  "$TU_DBS_EXCLUDED_TASK1_RUNS" \
  "$TU_DBS_EXCLUDED_TASK3_RUNS"
