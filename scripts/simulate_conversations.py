import json
from pathlib import Path
from datetime import datetime
import argparse
import ollama

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]

# data/processed/
PROCESSED_DIR = ROOT / "data" / "processed"
POSTS_FILENAME = "posts_highconf_nohoney_majority_sev4.json"

# base output dir; one subfolder per subreddit
CONV_ROOT = PROCESSED_DIR / "conversations_nonbinary"
CONV_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
OLLAMA_MODEL = "llama3"  # change to any installed Ollama model

# Conversation config
MAX_TURNS = 6
STOP_TOKEN = "[end-of-dialogue]"

# Per-subreddit demographic descriptions
SUBREDDIT_DEMOGRAPHICS = {
    "NonBinary": "non-binary person posting in r/NonBinary",
    "TwoXChromosomes": "woman posting in r/TwoXChromosomes",
    "AskMen": "man posting in r/AskMen",
    "Daddit": "dad posting in r/Daddit",
    "Mommit": "mom posting in r/Mommit",
}

DEFAULT_DEMOGRAPHIC = "person posting on Reddit"


def get_demographic_for_subreddit(subreddit: str) -> str:
    return SUBREDDIT_DEMOGRAPHICS.get(subreddit, DEFAULT_DEMOGRAPHIC)


def call_llm_ollama(messages):
    """
    Call Ollama local model with chat format.
    """
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return "[Error generating response]"


def make_user_turn_prompt(demographic: str, original_post: str, history: list):
    """
    Simulates the user's next turn as a casual, emotionally expressive human.
    """
    return [
        {
            "role": "system",
            "content": (
                f"You are roleplaying a {demographic} seeking social support. "
                "Stay in character, don't mention being an AI, and don't give advice. "
                "Respond like a real person would: casually, honestly, even emotionally. "
                "You can include typos, jokes, sarcasm, or personal anecdotes. "
                "You're not trying to be perfect—just real. Express how you're feeling, or ask a follow-up."
            ),
        },
        {
            "role": "system",
            "content": f"Original concern:\n{original_post}",
        },
        *history,
        {
            "role": "system",
            "content": "Now continue the conversation by writing the next message from the user.",
        },
    ]


def make_agent_turn_prompt(demographic: str, history: list):
    """
    Prepares the assistant's next turn with supportive tone and reflective listening.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a compassionate, supportive assistant. "
                f"You are talking to someone from this demographic: {demographic}. "
                "Use empathy, reflective listening, and gentle suggestions. "
                "Validate their concerns. Be emotionally warm, casual, and avoid being overly formal. "
                f"When the user seems satisfied or there's nothing more to say, append this token: {STOP_TOKEN}"
            ),
        },
        *history,
    ]


def simulate_conversation(post: dict, demographic: str, max_turns: int = MAX_TURNS):
    original_text = post.get("body") or post.get("title") or ""
    history = [{"role": "user", "content": original_text}]

    for _ in range(max_turns):
        # assistant turn
        agent_prompt = make_agent_turn_prompt(demographic, history)
        agent_resp = call_llm_ollama(agent_prompt)
        history.append({"role": "assistant", "content": agent_resp})

        if STOP_TOKEN in agent_resp:
            break

        # user turn
        user_prompt = make_user_turn_prompt(demographic, original_text, history)
        user_resp = call_llm_ollama(user_prompt)
        history.append({"role": "user", "content": user_resp})

    return history


def process_subreddit(subreddit: str):
    """
    Load the posts for a single subreddit, generate conversations,
    and write them to the appropriate folder.
    """
    input_path = PROCESSED_DIR / subreddit / POSTS_FILENAME
    if not input_path.exists():
        print(f"❗ Subreddit '{subreddit}' has no {POSTS_FILENAME} at {input_path}, skipping.")
        return

    demographic = get_demographic_for_subreddit(subreddit)

    # output dir for this subreddit
    output_dir = CONV_ROOT / subreddit
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = json.loads(input_path.read_text())
    print(f"📥 [{subreddit}] Loaded {len(posts)} posts from {input_path}")

    for i, post in enumerate(posts, start=1):
        out_path = output_dir / f"conv_{post['post_id']}.json"
        if out_path.exists():
            print(f"[{subreddit}] [{i}/{len(posts)}] ⏩ Skipping {post['post_id']} (already exists)")
            continue

        convo = simulate_conversation(post, demographic)

        convo_obj = {
            "post_id": post["post_id"],
            "title": post.get("title", ""),
            "demographic": demographic,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "turns": convo,
            "model": OLLAMA_MODEL,
            "subreddit": subreddit,
        }

        out_path.write_text(json.dumps(convo_obj, indent=2))
        print(f"[{subreddit}] [{i}/{len(posts)}] ✅ Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Simulate support conversations for Reddit posts.")
    parser.add_argument(
        "--subreddit",
        help=(
            "Name of subreddit folder under data/processed "
            "(e.g., NonBinary, AskMen, Daddit, Mommit, TwoXChromosomes). "
            "If omitted, process all subreddits that have a "
            "posts_highconf_nohoney_majority_sev4.json file."
        ),
    )
    args = parser.parse_args()

    if args.subreddit:
        process_subreddit(args.subreddit)
    else:
        # all subdirs in data/processed/ that have the posts file
        subreddits = [
            d.name
            for d in PROCESSED_DIR.iterdir()
            if d.is_dir() and (d / POSTS_FILENAME).exists()
        ]

        if not subreddits:
            print("❌ No subreddit folders with filtered posts found in data/processed/")
            return

        print(f"🔍 Found subreddits: {', '.join(subreddits)}")
        for sub in subreddits:
            process_subreddit(sub)


if __name__ == "__main__":
    main()