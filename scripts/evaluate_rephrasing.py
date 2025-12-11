import json
import argparse
from pathlib import Path
import ollama

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
EVALUATION_DIR = ROOT / "data" / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
OLLAMA_MODEL = "llama3.1:8b"

# ---------- EVALUATION CRITERIA ----------
EVALUATION_PROMPT = """
You are an expert in text analysis and evaluation. Your task is to evaluate a rephrased message based on how well it represents an original post.

Please evaluate the rephrased message based on the following criteria:

1.  **Faithfulness:** Does the rephrased message accurately reflect the core meaning and intent of the original post? (Scale: 1-5, where 1 is not faithful at all and 5 is very faithful)
2.  **Conciseness:** Is the rephrased message shorter and more to the point than the original post? (Scale: 1-5, where 1 is not concise at all and 5 is very concise)
3.  **Naturalness:** Does the rephrased message sound like something a person would naturally say in a conversation? (Scale: 1-5, where 1 is not natural at all and 5 is very natural)
4.  **Completeness:** Does the rephrased message capture all the key information from the original post? (Scale: 1-5, where 1 is not complete at all and 5 is very complete)

**Original Post:**
---
{original_post}
---

**Rephrased Message:**
---
{rephrased_message}
---

Please provide your evaluation in JSON format with the following structure:
{
  "faithfulness": <score>,
  "conciseness": <score>,
  "naturalness": <score>,
  "completeness": <score>,
  "reasoning": "<your reasoning for the scores>"
}
"""


def get_original_post(subreddit: str, post_id: str) -> str:
    """Loads the original post from the raw data."""
    raw_post_file = RAW_DIR / subreddit / "posts.json"
    if not raw_post_file.exists():
        return ""

    posts = json.loads(raw_post_file.read_text())
    for post in posts:
        if post["post_id"] == post_id:
            return post.get("body") or post.get("title") or ""
    return ""


def evaluate_rephrasing(original_post: str, rephrased_message: str) -> dict:
    """Uses an LLM to evaluate the rephrased message."""
    if not original_post or not rephrased_message:
        return {}

    prompt = EVALUATION_PROMPT.format(
        original_post=original_post,
        rephrased_message=rephrased_message,
    )

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        print(f"❌ Error evaluating rephrasing: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rephrased messages using an LLM judge."
    )
    parser.add_argument(
        "--subreddit",
        help="Name of subreddit folder (e.g., NonBinary, AskMen). If omitted, process all.",
    )
    args = parser.parse_args()

    subreddits = []
    if args.subreddit:
        subreddits.append(args.subreddit)
    else:
        subreddits = [
            d.name
            for d in PROCESSED_DIR.iterdir()
            if d.is_dir() and (d / "conversations").exists()
        ]

    for subreddit in subreddits:
        print(f"Processing subreddit: {subreddit}")
        conv_dir = PROCESSED_DIR / subreddit / "conversations"
        if not conv_dir.exists():
            continue

        for conv_file in conv_dir.glob("*.json"):
            print(f"  Processing conversation: {conv_file.name}")
            conv_data = json.loads(conv_file.read_text())
            post_id = conv_data["post_id"]
            opening_message = conv_data["opening_message"]

            original_post = get_original_post(subreddit, post_id)
            if not original_post:
                print(f"    ❌ Could not find original post for {post_id}")
                continue

            evaluation = evaluate_rephrasing(original_post, opening_message)
            if not evaluation:
                continue

            evaluation_file = EVALUATION_DIR / subreddit / f"eval_{post_id}.json"
            evaluation_file.parent.mkdir(parents=True, exist_ok=True)
            evaluation_file.write_text(json.dumps(evaluation, indent=2))
            print(f"    ✅ Saved evaluation to {evaluation_file}")


if __name__ == "__main__":
    main()
