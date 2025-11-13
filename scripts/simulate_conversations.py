import json
import re
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ollama

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
POSTS_FILENAME = "posts_highconf_nohoney_majority_sev4.json"
CONV_ROOT = PROCESSED_DIR / "conversations"
CONV_ROOT.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
OLLAMA_MODEL = "llama3"
MAX_EXCHANGES = 8

# ---------- DEMOGRAPHICS ----------
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

# ---------- LOW-LEVEL LLM CALL ----------
def call_llm_ollama(messages):
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.7},
        )
        return (response["message"]["content"] or "").strip()
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return ""

# ---------- INITIAL PROCESSING ----------
def build_shard_extraction_messages(post_text: str):
    system_prompt = (
        "You are a psychological analyst. Your task is to read the following Reddit post and break it down into a list of core 'shards' of information. Each shard should represent a distinct theme, feeling, event, or belief expressed by the author.\n"
        "- Extract between 5 and 10 shards.\n"
        "- Each shard should be a short, self-contained statement.\n"
        "- Phrase the shards from the author's perspective (e.g., \"I feel like my partner doesn't respect me\").\n"
        "- Present the shards as a numbered list."
    )
    user_prompt = f"Reddit Post:\n---\n{post_text}\n---\n\nNumbered list of shards:"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def extract_shards_from_post(post_text: str) -> list[str]:
    if not post_text.strip(): return []
    for _ in range(2):
        messages = build_shard_extraction_messages(post_text)
        response = call_llm_ollama(messages)
        shards = re.findall(r"^\s*\d+\.\s+(.*)", response, re.MULTILINE)
        if shards: return [s.strip() for s in shards]
    print("❌ Shard extraction failed after 2 attempts.")
    return []

def build_opening_message_messages(post_text: str):
    system_prompt = (
        "Read the following Reddit post. Your task is to write a short, one-sentence opening message that a user might say to a therapist or a supportive friend to start a conversation about the problem described in the post.\n"
        "- The message should be in the first person ('I', 'me').\n"
        "- It should be a natural, conversational opening, not a perfect summary.\n"
        "- Do not include any pleasantries like 'Hi' or 'Hello'."
    )
    user_prompt = f"Reddit Post:\n---\n{post_text}\n---\n\nOpening message:"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def generate_opening_message(post_text: str) -> str:
    if not post_text.strip(): return ""
    for _ in range(2):
        messages = build_opening_message_messages(post_text)
        response = call_llm_ollama(messages)
        if response: return response
    print("❌ Opening message generation failed.")
    return ""

# ---------- PROMPT BUILDERS ----------
def build_assistant_messages(demographic: str, history: list):
    system_prompt = (
        "You are a warm, compassionate online support partner having a casual chat.\n"
        f"- You are talking to a {demographic}.\n"
        "- This is not therapy or medical advice; you are just a kind, supportive peer.\n"
        "- Use empathy and reflective listening. Keep replies short (2–6 sentences).\n"
        "- Often end with ONE gentle, open question to invite another reply.\n"
        "- Do NOT mention being an AI or a chatbot.\n\n"
        "Base your reply ONLY on the recent conversation history below. Continue the conversation with your next reply."
    )
    return [{"role": "system", "content": system_prompt}, *history[-8:]]

def build_shard_selection_messages(history: list, available_shards: list):
    shards_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(available_shards)])
    therapist_last_message = history[-1]['content']
    system_prompt = (
        "You are a helpful assistant. Your task is to select the most relevant 'shard' for a user to discuss next in a therapy session.\n"
        "Based on the therapist's last message and the user's available shards of concern, which shard is the most logical one to bring up now?\n"
        "Respond with ONLY the number of the shard."
    )
    user_prompt = (
        f"Therapist's last message:\n---\n{therapist_last_message}\n---\n\n"
        f"User's available shards of concern:\n---\n{shards_text}\n---\n\n"
        "Respond with the single best shard number to discuss next."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def build_response_generation_messages(demographic: str, history: list, selected_shard: str):
    therapist_last_message = history[-1]['content']
    system_prompt = (
        "You are roleplaying a person in a therapy session. Your demographic is: {demographic}.\n"
        "You are responding to your therapist. Your response should be based on the 'core feeling' you are currently focused on.\n"
        "Keep your reply short and natural (1-4 sentences). Respond in the first person ('I', 'me')."
    )
    user_prompt = (
        f"Your therapist just said:\n---\n{therapist_last_message}\n---\n\n"
        f"The core feeling/thought you are focused on right now is:\n---\n{selected_shard}\n---\n\n"
        "Write your response to the therapist."
    )
    return [{"role": "system", "content": system_prompt.format(demographic=demographic)}, {"role": "user", "content": user_prompt}]

def build_controller_messages(history: list):
    transcript = "\n".join([f"{t['role']}: {t['content']}" for t in history[-6:]])
    system_prompt = (
        "You are a conversation controller. Decide if the conversation should CONTINUE or END.\n"
        "- CONTINUE if the user has more to say.\n"
        "- END if the conversation is resolved or winding down.\n"
        "Respond with EXACTLY one word: CONTINUE or END."
    )
    user_prompt = f"Conversation:\n{transcript}\n\nShould the conversation CONTINUE or END?"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

# ---------- AGENT REPLY HELPERS ----------
def generate_assistant_reply(demographic, history):
    for _ in range(2):
        msgs = build_assistant_messages(demographic, history)
        resp = call_llm_ollama(msgs)
        if resp: return resp
    print("⚠️ Assistant failed to generate a reply.")
    return None

def generate_user_reply(demographic, history, available_shards):
    # Step 1: Select a shard
    shard_idx = -1
    for _ in range(2):
        selection_msgs = build_shard_selection_messages(history, available_shards)
        resp = call_llm_ollama(selection_msgs)
        match = re.search(r'\d+', resp)
        if match:
            try:
                num = int(match.group(0)) - 1
                if 0 <= num < len(available_shards):
                    shard_idx = num
                    break
            except (ValueError, IndexError):
                continue
    if shard_idx == -1:
        print("⚠️ User failed to select a valid shard.")
        return None, None

    # Step 2: Generate a response based on the selected shard
    selected_shard = available_shards[shard_idx]
    for _ in range(2):
        response_msgs = build_response_generation_messages(demographic, history, selected_shard)
        resp = call_llm_ollama(response_msgs)
        if resp:
            return resp, shard_idx
            
    print("⚠️ User failed to generate a response.")
    return None, None

# ---------- CONTROLLER AGENT ----------
def controller_decision(history: list) -> str:
    msgs = build_controller_messages(history)
    resp = call_llm_ollama(msgs).upper()
    if "END" in resp: return "END"
    if "CONTINUE" in resp: return "CONTINUE"
    return "END" if len(history) // 2 >= MAX_EXCHANGES else "CONTINUE"

# ---------- MAIN SIMULATION LOOP ----------
def simulate_conversation(post: dict, demographic: str, max_exchanges: int = MAX_EXCHANGES):
    original_text = post.get("body") or post.get("title") or ""
    if not original_text.strip(): return None

    shards = extract_shards_from_post(original_text)
    if not shards: return None

    opening_message = generate_opening_message(original_text)
    if not opening_message: return None
    
    available_shards = list(shards)
    history = [{"role": "user", "content": opening_message}]
    used_shards = []

    for _ in range(max_exchanges):
        if not available_shards:
            print("ℹ️ No more shards to discuss.")
            break

        assistant_resp = generate_assistant_reply(demographic, history)
        if not assistant_resp: break
        history.append({"role": "assistant", "content": assistant_resp})

        user_resp, used_shard_idx = generate_user_reply(demographic, history, available_shards)
        if user_resp is None: break
        history.append({"role": "user", "content": user_resp})
        
        used_shard_text = available_shards.pop(used_shard_idx)
        used_shards.append(used_shard_text)

        if controller_decision(history) == "END":
            print("ℹ️ Controller decided to end conversation.")
            break

    return [t for t in history if t.get("content", "").strip()], shards, used_shards, opening_message

# ---------- IO WRAPPER ----------
def process_subreddit(subreddit: str, limit: int | None = None):
    input_path = PROCESSED_DIR / subreddit / POSTS_FILENAME
    if not input_path.exists():
        print(f"❗ No posts file at {input_path}, skipping.")
        return

    demographic = get_demographic_for_subreddit(subreddit)
    output_dir = CONV_ROOT / subreddit
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = json.loads(input_path.read_text())
    if limit: posts = posts[:limit]
    print(f"📥 [{subreddit}] Loaded {len(posts)} posts.")

    for i, post in enumerate(posts, start=1):
        out_path = output_dir / f"conv_{post['post_id']}.json"
        if out_path.exists():
            print(f"[{subreddit}] [{i}/{len(posts)}] ⏩ Skipping {post['post_id']}")
            continue

        print(f"[{subreddit}] [{i}/{len(posts)}] 🧠 Simulating {post['post_id']}...")
        result = simulate_conversation(post, demographic)
        if not result:
            print(f"[{subreddit}] [{i}/{len(posts)}] ❌ Simulation failed for {post['post_id']}")
            continue
        
        convo, all_shards, used_shards, opening_message = result
        convo_obj = {
            "post_id": post["post_id"], "title": post.get("title", ""), "demographic": demographic,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "opening_message": opening_message,
            "shards": {"all": all_shards, "used": used_shards},
            "turns": convo, "model": OLLAMA_MODEL, "subreddit": subreddit,
        }
        out_path.write_text(json.dumps(convo_obj, indent=2))
        print(f"[{subreddit}] [{i}/{len(posts)}] ✅ Wrote {out_path}")

# ---------- CLI ENTRYPOINT ----------
def main():
    parser = argparse.ArgumentParser(description="Simulate support conversations for Reddit posts.")
    parser.add_argument("--subreddit", help="Name of subreddit folder (e.g., NonBinary, AskMen). If omitted, process all.")
    parser.add_argument("--limit", type=int, help="Limit the number of posts to process per subreddit.")
    args = parser.parse_args()

    if args.subreddit:
        process_subreddit(args.subreddit, limit=args.limit)
    else:
        subreddits = [d.name for d in PROCESSED_DIR.iterdir() if d.is_dir() and (d / POSTS_FILENAME).exists()]
        if not subreddits:
            print("❌ No subreddit folders with filtered posts found.")
            return
        print(f"🔍 Found subreddits: {', '.join(subreddits)}")
        for sub in subreddits:
            process_subreddit(sub, limit=args.limit)

if __name__ == "__main__":
    main()
