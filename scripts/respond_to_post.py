import json
import re
from pathlib import Path
from datetime import datetime, timezone
import argparse
import time
import google.generativeai as genai

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
POSTS_FILENAME = "posts_highconf_nohoney_majority_sev4.json"

# ---------- MODEL CONFIG ----------
GEMINI_MODEL = "models/gemini-flash-latest"

# ---------- SLOW MODE CONFIG ----------
LAST_GEMINI_CALL_TIMESTAMP = 0
GEMINI_CALL_INTERVAL = 4  # seconds (to stay under 15 RPM)

# ---------- DEMOGRAPHICS ----------
SUBREDDIT_DEMOGRAPHICS = {
    "NonBinary": "non-binary person",
    "TwoXChromosomes": "woman",
    "AskMen": "man",
    "Daddit": "dad",
    "Mommit": "mom",
}
DEFAULT_DEMOGRAPHIC = "person"


def get_demographic_for_subreddit(subreddit: str) -> str:
    return SUBREDDIT_DEMOGRAPHICS.get(subreddit, DEFAULT_DEMOGRAPHIC)


# ---------- LOW-LEVEL LLM CALL ----------
def call_llm_gemini(messages, json_mode: bool = False):
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
        config = {"temperature": 0.7}
        if json_mode:
            config["response_mime_type"] = "application/json"

        gemini_messages = []
        for message in messages:
            if message["role"] == "system":
                gemini_messages.append({"role": "user", "parts": [message["content"]]})
            elif message["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [message["content"]]})
            else:
                gemini_messages.append({"role": message["role"], "parts": [message["content"]]})

        merged_messages = []
        for message in gemini_messages:
            if merged_messages and merged_messages[-1]["role"] == "user" and message["role"] == "user":
                merged_messages[-1]["parts"][-1] += "\n" + message["parts"][-1]
            else:
                merged_messages.append(message)


        response = model.generate_content(merged_messages, generation_config=config)
        LAST_GEMINI_CALL_TIMESTAMP = time.time()
        return response.text
    except Exception as e:
        print(f"❌ Gemini error: {e}")
        return ""


def call_llm(messages, model: str, json_mode: bool = False):
    return call_llm_gemini(messages, json_mode=json_mode)


# ---------- PROMPT BUILDERS ----------
def build_assistant_messages(demographic: str, history: list):
    system_prompt = (
        "You are having a conversation with a user who is looking for support. "
        "If something is not clear, you can ask the user to clarify what they need."
    )
    return [{"role": "system", "content": system_prompt}, *history]


def generate_assistant_reply(demographic, history):
    for _ in range(2):
        msgs = build_assistant_messages(demographic, history)
        resp = call_llm(msgs, model="gemini")
        if resp:
            return resp
    print("⚠️ Assistant failed to generate a reply.")
    return None


# ---------- MAIN RESPONSE LOGIC ----------
def respond_to_post(post: dict, demographic: str):
    original_text = post.get("body") or post.get("title") or ""
    if not original_text.strip():
        return None

    post_title = post.get("title", "")
    
    # The "history" is now just the full post
    history = [{"role": "user", "content": f"{post_title}\n\n{original_text}"}]

    assistant_resp = generate_assistant_reply(demographic, history)

    if not assistant_resp:
        return None

    return {
        "post_id": post["post_id"],
        "title": post_title,
        "demographic": demographic,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "turns": [
            {"role": "user", "content": f"{post_title}\n\n{original_text}"},
            {"role": "assistant", "content": assistant_resp}
        ],
        "model": GEMINI_MODEL,
        "subreddit": post.get("subreddit_name"),
    }


# ---------- IO WRAPPER ----------
def process_subreddit(subreddit: str, limit: int | None = None, post_id: str | None = None):
    input_path = PROCESSED_DIR / subreddit / POSTS_FILENAME
    if not input_path.exists():
        print(f"❗ No posts file at {input_path}, skipping.")
        return

    demographic = get_demographic_for_subreddit(subreddit)

    posts = json.loads(input_path.read_text())
    
    if post_id:
        posts = [p for p in posts if p["post_id"] == post_id]
        if not posts:
            print(f"❌ Post with ID '{post_id}' not found in {subreddit}.")
            return

    if limit:
        posts = posts[:limit]
    print(f"📥 [{subreddit}] Loaded {len(posts)} posts.")

    for i, post in enumerate(posts, start=1):
        post_output_dir = PROCESSED_DIR / subreddit / post["post_id"]
        post_output_dir.mkdir(parents=True, exist_ok=True)
        out_path = post_output_dir / "whole_post_response.json"

        if out_path.exists():
            print(f"[{subreddit}] [{i}/{len(posts)}] ⏩ Skipping {post['post_id']}")
            continue

        print(f"[{subreddit}] [{i}/{len(posts)}] 🧠 Responding to {post['post_id']}...")
        result = respond_to_post(post, demographic)
        if not result:
            print(
                f"[{subreddit}] [{i}/{len(posts)}] ❌ Response generation failed for {post['post_id']}"
            )
            continue

        out_path.write_text(json.dumps(result, indent=2))
        print(f"[{subreddit}] [{i}/{len(posts)}] ✅ Wrote {out_path}")


# ---------- CLI ENTRYPOINT ----------
def main():
    parser = argparse.ArgumentParser(
        description="Generate an assistant response for the entire post."
    )
    parser.add_argument(
        "--subreddit",
        help="Name of subreddit folder (e.g., NonBinary, AskMen). If omitted, process all.",
    )
    parser.add_argument(
        "--limit", type=int, help="Limit the number of posts to process per subreddit."
    )
    parser.add_argument(
        "--post_id", help="The ID of the post to process."
    )
    args = parser.parse_args()

    if args.subreddit:
        process_subreddit(args.subreddit, limit=args.limit, post_id=args.post_id)
    else:
        subreddits = [
            d.name
            for d in PROCESSED_DIR.iterdir()
            if d.is_dir() and (d / POSTS_FILENAME).exists()
        ]
        if not subreddits:
            print("❌ No subreddit folders with filtered posts found.")
            return
        print(f"🔍 Found subreddits: {', '.join(subreddits)}")
        for sub in subreddits:
            process_subreddit(sub, limit=args.limit, post_id=args.post_id)


if __name__ == "__main__":
    main()
