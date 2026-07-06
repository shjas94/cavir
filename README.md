# CaVIR

**Cost-Aware Visual Reasoning with Learned Operation Necessity, Calibrated Abstention, and Mediation-Based Validation**

CaVIR teaches a vision–language model to decide **when a tool is worth using and which tool gives the best value for its cost**, to **abstain** when it is not confident enough, and to **causally validate** that a tool call actually contributed to the answer.

> See [`docs/CaVIR_Research_Proposal_v4.md`](docs/CaVIR_Research_Proposal_v4.md) for the full research proposal.

---

## Core Idea

Modern visual-reasoning models call tools (zoom-in, code execution, sketching) to solve hard problems, but they do not learn *whether* a given tool is actually needed. CaVIR learns a single score, **V-VON** (Value-based Visual Operation Necessity), that estimates the net benefit of a tool minus its cost, and uses it to:

- **Route** — pick the tool with the highest positive V-VON, otherwise skip.
- **Abstain** — hold back the final answer when the calibrated correctness probability is too low.
- **Validate** — check, after training, whether tool calls causally contribute to the answer.

---

## Roadmap

CaVIR is built as a four-stage training pipeline. Only **cold-start dataset construction** is implemented so far; the remaining stages are in progress.

- [x] **Cold-start dataset construction** — collect visual-reasoning traces with tool calls (see below).
- [ ] **Stage A — Text warm-up SFT** *(in progress)*
- [ ] **Stage B — Visual-tool SFT** *(in progress)*
- [ ] **Stage C — Necessity-score learning (CANE-Predictor)** *(in progress)*
- [ ] **Stage D — Reinforcement learning (Dr.GRPO + AXPO + ESPO)** *(in progress)*
- [ ] **Evaluation & causal-mediation analysis** *(in progress)*

---

## Cold-Start Dataset Construction

The only implemented module. It runs a fixed data-generator VLM as a visual-reasoning agent over a VQA dataset, letting it reason step by step and call tools, and records the resulting traces as JSONL for downstream supervised fine-tuning.

**Location:** [`dataset_construction/`](dataset_construction/)

- `host.py` — agent orchestrator: drives the VLM through reasoning/tool-calling cycles via MCP.
- `dataset.py` — dataset loading and image pre-processing.
- `run.sh` — entry point.
- `server/` — MCP tool servers:
  - **Code execution** (`code_execution_tool.py`): `run_code` in a sandboxed interpreter.
  - **Image processing** (`image_processing_tool.py`): `crop_zoomin`, `mark_dots`, `draw_bbox`, `crop_ocr_image`, `imaginate_editing`.

Supported source datasets: `docvqa`, `chartqa_1`, `chartqa_2`, `infographics_vqa`.

### Usage

```bash
cd dataset_construction
./run.sh                       # defaults to the docvqa dataset
./run.sh chartqa_1             # choose a dataset
```

Outputs (traces and intermediate tool results) are written to `dataset_construction/outputs/<dataset>_<timestamp>/`.

### Configuration

Set these in `dataset_construction/.env` (loaded automatically by `run.sh`):

| Variable | Description |
| --- | --- |
| `DIRECT_BASE_URL` | Base URL of the OpenAI-compatible VLM server (e.g. vLLM). |
| `DIRECT_API_KEY` | API key for that server. |
| `MODEL_NAME` | Data-generator model to serve as the agent. |

`run.sh` also accepts `CONCURRENCY`, `SAMPLE`, and `OUTPUT_DIR` as environment overrides.

---

## Models

| Role | Model |
| --- | --- |
| Policy (trained) | Qwen3.5-4B |
| Scorer (trained) | Qwen3.5-2B |
| Data generator (fixed) | Qwen3.6-35B-A3B |
