# CaVIR

Code for a research project on how vision-language models use visual tools such as cropping and OCR while
reasoning about an image. It covers generating tool-using training data, supervised fine-tuning, training a
value model, and evaluation on public benchmarks.

## Layout

| Folder | Contents |
| --- | --- |
| `dataset_construction/` | Tool-using rollouts from a model served with vLLM (`host.py`, MCP tool servers in `server/`) and building of the SFT data |
| `sft/` | LLaMA-Factory configs and multi-node launchers for supervised fine-tuning, and evaluation helpers |
| `value_head/` | Training data, training and evaluation of the value model |
| `benchmark_eval/` | Evaluation on public benchmarks through VLMEvalKit |
| `scripts/` | Copying code and models to the second node |

## Setup

- Python environments: a data environment (vLLM client, MCP, transformers) for `dataset_construction/`,
  `value_head/` and `benchmark_eval/`, and a training environment with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
  installed from a checkout at `sft/LlamaFactory/` (not part of this repository).
- `dataset_construction/.env`: copy `.env.example` and set the served model's URL, API key and name.
- `cluster.env` at the repository root (two-node runs only): `NODE0_ADDR`, `NODE1_SSH`, `NODE1_HOME`,
  `NODE1_CONTAINER`, `NET_IFNAME`. Both `.env` files are git-ignored.
- Dataset and model paths (Hugging Face cache locations in `dataset_construction/host.py` and `dataset.py`) and the
  default interpreter paths in the shell scripts are machine-specific; adapt them before running.

## Running

Data generation (from `dataset_construction/`, with the teacher served):

    ./run.sh docvqa 100                      # rollouts: DATASET [SAMPLE_COUNT]; IDS_FILE=<file> restricts the samples
    ./make_coldstart_v2.sh docvqa chartqa_1  # filter, reformat, build, quality filter, merge into LLaMA-Factory data

Supervised fine-tuning (two nodes; the same command with `run_node1_fsdp.sh` on the second node):

    CONFIG=../configs/qwen35_full_cavir_v2_noarith_fixtools.yaml sft/scripts/run_node0_fsdp.sh

Value head: `value_head/decision_points.py`, `branch_rollout.py`, `train_scorer.py` and `evaluate_scorer.py`, each
documented in its header.

Benchmark evaluation: see `benchmark_eval/README.md`.
