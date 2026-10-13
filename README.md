# SOFT

This archive contains the test-time training and evaluation path for SOFT. The frozen speech language model supplies a full-history teacher; a short-history student updates session fast weights and writes its own response back into the dialogue.

## Setup

Use Python 3.12, CUDA 12.8, and a GPU with about 80 GB or more memory. The archive includes the minimal ms-swift 4.1 runtime used by this pipeline under its bundled Apache 2.0 license. The recorded environment used PyTorch 2.9.0 and Transformers 4.57.6.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are not included because of the archive-size limit. `--model` accepts a local ParaBridge-processed Qwen3-Omni checkpoint.

## Data

`data/sessions.json` contains two synthetic sessions with 24 kHz mono audio and relative paths only. Check the archive without loading a model:

```bash
python train.py --check
```

Custom sessions use the same fields: `session_id`, `scenario`, `axis`, `key_turns`, and ordered turns containing `turn`, `text`, `condition`, `state`, and a relative `audio` path.

## Run

```bash
python train.py \
  --model MODEL_DIR \
  --data data/sessions.json \
  --methods short,full,memo,soft,soft_fc \
  --max-new-tokens 1024 \
  --output results
```

The defaults reproduce the primary configuration: layers 38, 40, 42, 44, and 46; rank 4; three Adam steps per turn; learning rate 0.002 with 0.92 turn decay; 0.85 factor carry decay; symmetric KL at temperature 1.2; residual cap 1.5; and a current-plus-two-turn student window. The default generation cap is 512; the command above uses the paper's 1024-token evaluation cap. `soft_fc` changes only the final decode to full history.

For another compatible backbone, also set `--model-type`, `--template-type`, `--hidden-size`, `--layers`, `--forward-path`, and `--layer-path` for its layout.

## Evaluate

The evaluator sends one anonymous response at a time to an OpenAI-compatible judge and uses the session's key turns by default. Run it once per judge model; `score` is Overall and `focus` is Control in the paper.

```bash
JUDGE_BASE_URL=ENDPOINT \
JUDGE_API_KEY=KEY \
JUDGE_MODEL=MODEL \
python eval.py --results results --focus auto --output scores.json
```

Use `python eval.py --results results --check` to validate result files without making requests.
