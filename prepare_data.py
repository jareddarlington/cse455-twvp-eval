"""Rebuild data/{test,dev}.jsonl + images from the public HF dataset, row-for-row identical to the
JSONL the report evals used (only the image paths differ: they point at ./data/images).

  python prepare_data.py [--splits test dev] [--out data]

Needs: huggingface_hub, pyarrow.
"""
import argparse
import json
import os

REPO = "jareddarlington/cse455-twvp-objects365"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["test", "dev"])
    ap.add_argument("--out", default="data")
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    out = os.path.abspath(a.out)
    for split in a.splits:
        pq_path = hf_hub_download(REPO, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        rows = pq.read_table(pq_path).to_pylist()
        img_dir = os.path.join(out, "images", split)
        os.makedirs(img_dir, exist_ok=True)
        with open(os.path.join(out, f"{split}.jsonl"), "w") as f:
            for r in rows:
                img = os.path.join(img_dir, r["image_file"])
                if not os.path.exists(img):
                    with open(img, "wb") as g:
                        g.write(r["image"]["bytes"])
                msgs = []
                for m in r["messages"]:
                    c = [{"type": "image", "image": img} if x["type"] == "image" else {"type": "text", "text": x["text"]}
                         for x in m["content"]]
                    if m["role"] == "assistant" and len(c) == 1:
                        c = c[0]["text"]  # original JSONL stores the assistant turn as a plain string
                    msgs.append({"role": m["role"], "content": c})
                row = {"id": r["id"], "source": r["source"], "task": r["task"], "split": r["split"], "image": img,
                       "width": r["width"], "height": r["height"], "category": r["category"], "answer": r["answer"],
                       "gt_boxes": r["gt_boxes"], "messages": msgs}
                f.write(json.dumps(row) + "\n")
        print(f"{split}: {len(rows)} rows -> {out}/{split}.jsonl", flush=True)


if __name__ == "__main__":
    main()
