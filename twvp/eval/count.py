"""Count accuracy (+ box F1 from the thinking) of a model on dev/test JSONL, with vLLM.

  python -m twvp.eval.count --model RUN_DIR/final --data datasets/objects365/dev.jsonl [--limit N]
         [--wandb-run-id ID --step S]      # attach metrics to a training run's W&B page

Greedy decoding, thinking on (generation prompt opens <think>), images capped like training.
"""
import argparse
import json
import os

from twvp.format import box_f1, parse_boxes, parse_count


def metrics(rows, preds, prefix):
    n = len(rows)
    # only a closed thinking block has a final answer; a truncated one counts as unparsed
    pc = [parse_count(p) if "</think>" in p else None for p in preds]
    ok = [c is not None for c in pc]
    ex = [c == r["answer"] for c, r in zip(pc, rows)]
    ob1 = [c is not None and abs(c - r["answer"]) <= 1 for c, r in zip(pc, rows)]
    ae = [abs(c - r["answer"]) for c, r in zip(pc, rows) if c is not None]
    f1s = []
    for p, r in zip(preds, rows):
        boxes = [b for _, bs in parse_boxes(p) for b in bs]
        if r["answer"] > 0 or boxes:
            f1s.append(box_f1(boxes, r["gt_boxes"])[2])
    return {f"{prefix}/count_acc": sum(ex) / n, f"{prefix}/off_by_one": sum(ob1) / n,
            f"{prefix}/mae": sum(ae) / max(1, len(ae)), f"{prefix}/parse_rate": sum(ok) / n,
            f"{prefix}/box_f1@0.5": sum(f1s) / max(1, len(f1s)),
            f"{prefix}/think_closed": sum("</think>" in p for p in preds) / n,
            f"{prefix}/zero_acc": _acc([e for e, r in zip(ex, rows) if r["answer"] == 0]),
            f"{prefix}/n": n}


def _vllm_msg(m):
    """Dataset rows use {"type": "image", "image": path}; vLLM's chat API wants a PIL image part."""
    from PIL import Image
    c = [{"type": "image_pil", "image_pil": Image.open(x["image"]).convert("RGB")} if x["type"] == "image" else x
         for x in m["content"]]
    return {"role": m["role"], "content": c}


def _acc(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="datasets/objects365/dev.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--max-tokens", type=int, default=6144)   # base model thinks long; SFT model ~300
    ap.add_argument("--out", default=None)
    ap.add_argument("--wandb-run-id", default=None)
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.data)]
    rows = rows[: a.limit] if a.limit else rows
    split = os.path.basename(a.data).split(".")[0]
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, max_model_len=a.max_tokens + 2048, limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=0.85,
              seed=0, enforce_eager=True, mm_processor_kwargs={"max_pixels": a.max_pixels})
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
    msgs = [[_vllm_msg(m) for m in r["messages"] if m["role"] == "user"] for r in rows]
    outs = llm.chat(msgs, sp, use_tqdm=True)
    preds = [o.outputs[0].text for o in outs]
    # vLLM returns only the generated part; the prompt already opened <think>
    preds = [p if "<think>" in p else "<think>\n" + p for p in preds]
    m = metrics(rows, preds, split)
    print(json.dumps(m, indent=1), flush=True)

    out = a.out
    if not out:  # next to the checkpoint if we own it, else the runner's scratch
        own = os.access(a.model, os.W_OK) and "/hub/models--" not in os.path.abspath(a.model)  # never the HF cache
        d = a.model if own else os.environ.get("TWVP_RUNS", ".")
        tag = "" if own else ("base" if "/hub/models--" in os.path.abspath(a.model)
                              else os.path.basename(os.path.normpath(a.model))) + "_"
        os.makedirs(d, exist_ok=True)
        out = os.path.join(d, f"eval_{tag}{split}.jsonl")
    with open(out, "w") as f:
        for r, p, c in zip(rows, preds, [parse_count(p) if "</think>" in p else None for p in preds]):
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "pred": c, "text": p}) + "\n")
    json.dump(m, open(out.replace(".jsonl", "_metrics.json"), "w"), indent=1)
    if a.wandb_run_id:
        import wandb
        run = wandb.init(id=a.wandb_run_id, resume="allow")
        run.log({**m, **({"train/global_step": a.step} if a.step is not None else {})})
        tbl = wandb.Table(columns=["id", "answer", "pred", "text"])
        for r, p in list(zip(rows, preds))[:50]:
            tbl.add_data(r["id"], r["answer"], parse_count(p), p[-1500:])
        run.log({f"{split}/samples": tbl})
        run.finish()


if __name__ == "__main__":
    main()
