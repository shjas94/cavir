# benchmark_eval

Evaluates models served behind an OpenAI-compatible API on public benchmarks through VLMEvalKit, with tools.

## Setup

- A Python environment with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) importable (on `PYTHONPATH`),
  for `prepare.py` and `score.py`. `stubs/` holds a placeholder for `decord`, which VLMEvalKit imports for video
  benchmarks; add it to `PYTHONPATH` where `decord` cannot be installed.
- Benchmark files in `$LMUData`, as VLMEvalKit expects them.
- The model to evaluate served behind an OpenAI-compatible API that accepts `continue_final_message` and `bad_words`
  (vLLM). The API key is read from `dataset_construction/.env`.

## Running

    P=benchmark_eval/outputs/problems
    python benchmark_eval/prepare.py --out $P
    O=benchmark_eval/outputs/<model>/<dataset>
    python benchmark_eval/rollout.py natural --problems $P/<dataset>.jsonl --out $O --base-url URL --model NAME --tokenizer PATH
    python benchmark_eval/rollout.py branch --out $O --base-url URL --model NAME --tokenizer PATH
    python benchmark_eval/rollout.py predictions --out $O
    python benchmark_eval/faithfulness.py --problems $P/<dataset>.jsonl --out $O
    python benchmark_eval/score.py --dataset <dataset> --out $O   # VStarBench: add --judge-base-url URL --judge-model NAME
    python benchmark_eval/analyze.py --out $O
