# CSE 455 Group 2: Objects365 counting eval

The exact eval used for the "Thinking with Visual Primitives" reports. `twvp/` is copied unchanged from the
main repo (commit d112501, the version the report evals ran).

## Run

```bash
pip install -r requirements.txt
python prepare_data.py              # test (952) + dev (466) from the public HF dataset -> data/
bash run_eval.sh Qwen/Qwen3.5-4B test
bash run_eval.sh /path/to/your/checkpoint test
```

Output: `results/<model>_<split>.jsonl` (one row per question) and `..._metrics.json`.

## Settings (do not change)

- vLLM, greedy (temperature 0), thinking on, max 6144 new tokens, `max_pixels` 1048576, `enforce_eager`, seed 0
- Count parsed with `twvp.format.parse_count`; an answer with no `</think>` counts as wrong
- Metric in the reports: `test/count_acc`

## Report numbers (test, 952 questions)

| model | count_acc | off_by_one | finished (`think_closed`) |
|---|---|---|---|
| Qwen3.5-4B base | 45.3% | 52.6% | 58.1% |
| Objects365 fine-tune (o365-v1) | 64.9% | 84.1% | 100% |

Data: https://huggingface.co/datasets/jareddarlington/cse455-twvp-objects365
