import json
import argparse
from pathlib import Path
import time
import google.generativeai as genai

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"
# EVALUATION_DIR is now dynamically created per post
# EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
GEMINI_MODEL = "models/gemini-flash-latest"

# ---------- SLOW MODE CONFIG ----------
LAST_GEMINI_CALL_TIMESTAMP = 0
GEMINI_CALL_INTERVAL = 4  # seconds (to stay under 15 RPM)

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
{{
  "faithfulness": <score>,
  "conciseness": <score>,
  "naturalness": <score>,
  "completeness": <score>,
  "reasoning": "<your reasoning for the scores>"
}}
"""


def get_original_post(subreddit: str, post_id: str) -> str:
    """Loads the original post from the raw data."""
    raw_post_file = RAW_DIR / subreddit / "posts.json"
    if not raw_post_file.exists():
        print(f"❗ Raw posts file not found for {subreddit} at {raw_post_file}")
        return ""

    posts = json.loads(raw_post_file.read_text())
    for post in posts:
        if post["post_id"] == post_id:
            return post.get("body") or post.get("title") or ""
    print(f"❗ Original post with ID {post_id} not found in {raw_post_file}")
    return ""


def call_llm_gemini(prompt: str) -> dict:
    """Uses Gemini to evaluate the rephrased message."""
    global LAST_GEMINI_CALL_TIMESTAMP
    try:
        # --- SLOW MODE ---
        elapsed_time = time.time() - LAST_GEMINI_CALL_TIMESTAMP
        if elapsed_time < GEMINI_CALL_INTERVAL:
            sleep_time = GEMINI_CALL_INTERVAL - elapsed_time
            print(f"--- Slowing down, sleeping for {sleep_time:.2f} seconds ---")
            time.sleep(sleep_time)
        # -----------------

        model = genai.GenerativeModel(GEMINI_MODEL)
        config = {"response_mime_type": "application/json"}
        response = model.generate_content(prompt, generation_config=config)
        LAST_GEMINI_CALL_TIMESTAMP = time.time()
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ Error evaluating rephrasing: {e}")
        return {}


def evaluate_rephrasing(original_post: str, rephrased_message: str) -> dict:
    """Uses an LLM to evaluate the rephrased message."""
    if not original_post or not rephrased_message:
        return {}

    prompt = EVALUATION_PROMPT.format(
        original_post=original_post,
        rephrased_message=rephrased_message,
    )

    return call_llm_gemini(prompt)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate rephrased messages using an LLM judge."
    )
    parser.add_argument(
        "--subreddit",
        help="Name of subreddit folder (e.g., NonBinary, AskMen). If omitted, process all.",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of conversations to process."
    )
    parser.add_argument(
        "--post_id", help="The ID of the post to process."
    )
    args = parser.parse_args()

    subreddits = []
    if args.subreddit:
        subreddits.append(args.subreddit)
    else:
        subreddits = [
            d.name
            for d in PROCESSED_DIR.iterdir()
            if d.is_dir() and (d / "conversations").exists() # This path will need to be updated
        ]

    for subreddit in subreddits:
        print(f"Processing subreddit: {subreddit}")
        
        # Find all post directories within the subreddit
        subreddit_processed_dir = PROCESSED_DIR / subreddit
        post_dirs = [d for d in subreddit_processed_dir.iterdir() if d.is_dir()]

        if args.post_id:
            post_dirs = [d for d in post_dirs if d.name == args.post_id]
            if not post_dirs:
                print(f"    ❌ Post with ID '{args.post_id}' not found in {subreddit}")
                continue

        if args.limit:
            post_dirs = post_dirs[: args.limit]

        for post_dir in post_dirs:
            post_id = post_dir.name
            conv_file = post_dir / "conversation.json"
            
            if not conv_file.exists():
                print(f"    ❌ Conversation file not found for post {post_id}, skipping.")
                continue

            print(f"  Processing conversation: {conv_file.name}")
            conv_data = json.loads(conv_file.read_text())
            

            original_post_full = get_original_post(subreddit, post_id)
            if not original_post_full:
                print(f"    ❌ Could not find original post for {post_id}")
                continue

            evaluations = []
            for i, turn in enumerate(conv_data["turns"]):
                if turn["role"] == "user":
                    rephrased_message = turn["content"]
                    original_text = ""
                    if i == 0:
                        original_text = original_post_full
                    elif "used_shard" in turn:
                        original_text = turn["used_shard"]["text"]

                    if not original_text:
                        continue

                    evaluation = evaluate_rephrasing(original_text, rephrased_message)
                    if evaluation:
                        evaluations.append(
                            {
                                "turn_index": i,
                                "rephrased_message": rephrased_message,
                                "original_text": original_text,
                                "evaluation": evaluation,
                            }
                        )

            if not evaluations:
                continue

            evaluation_file = post_dir / "rephrasing_evaluation.json"
            evaluation_file.parent.mkdir(parents=True, exist_ok=True)
            output_data = {"post_id": post_id, "evaluations": evaluations}
            evaluation_file.write_text(json.dumps(output_data, indent=2))
            print(f"    ✅ Saved evaluation to {evaluation_file}")


if __name__ == "__main__":
    main()
