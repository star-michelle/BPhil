import pandas as pd
import json
import os
from pathlib import Path

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

THRESHOLD = 0.5  # "mostly" severity 4


def filter_one_subreddit(posts_path: Path, ann_path: Path, output_path: Path, threshold: float = THRESHOLD):
    """
    Run your filtering logic for a single subreddit.
    This is basically your original code, wrapped in a function.
    """

    # --- load posts ---
    with open(posts_path, "r") as f:
        posts = json.load(f)
    posts_df = pd.DataFrame(posts)   # cols: post_id, title, body

    # --- load annotations ---
    ann_df = pd.read_csv(ann_path)

    # normalize string booleans
    # some CSVs might have "true"/"false", "TRUE"/"FALSE"
    ann_df["confidence"] = ann_df["confidence"].str.lower()
    ann_df["honey"] = ann_df["honey"].astype(str).str.lower()

    # 1) keep only high confidence
    ann_high = ann_df[ann_df["confidence"] == "high"].copy()

    # 2) drop honeyed (caught using AI)
    ann_clean = ann_high[ann_high["honey"] == "false"].copy()

    # group by post and compute severity 4 stats
    grouped = (
        ann_clean
        .groupby("post_id")
        .apply(
            lambda g: pd.Series({
                "n_ann": len(g),
                "n_sev4": (g["severity"] == 4).sum(),
                "p_sev4": (g["severity"] == 4).sum() / len(g)
            })
        )
        .reset_index()
    )

    # keep only posts that are mostly sev 4
    sev4_posts = grouped[grouped["p_sev4"] >= threshold]["post_id"]

    # join back to the real reddit posts
    result = posts_df.merge(sev4_posts, on="post_id", how="inner")

    # save minimal columns
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result[["post_id", "title", "body"]].to_json(output_path, orient="records", indent=2)

    print(f"[{posts_path}] wrote {len(result)} posts to {output_path}")
    return result


if __name__ == "__main__":
    """
    This part loops over EVERY subreddit folder in data/raw.
    For each folder that has posts.json and annotations.csv,
    it runs filter_one_subreddit on it.
    """

    for subdir in RAW_DIR.iterdir():
        if not subdir.is_dir():
            continue

        posts_path = subdir / "posts.json"
        ann_path = subdir / "annotations.csv"

        # skip if missing files
        if not posts_path.exists() or not ann_path.exists():
            print(f"Skipping {subdir.name}: missing posts.json or annotations.csv")
            continue

        subreddit = subdir.name
        output_path = PROCESSED_DIR / subreddit / "posts_highconf_nohoney_majority_sev4.json"

        filter_one_subreddit(posts_path, ann_path, output_path)