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
MAX_EXCHANGES = 20

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
def call_llm_ollama(messages, json_mode: bool = False):
    try:
        options = {"temperature": 0.7}
        if json_mode:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options=options,
                format="json"
            )
            return response["message"]["content"]
        else:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                options=options,
            )
            return (response["message"]["content"] or "").strip()
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return ""

# ---------- INITIAL PROCESSING ----------


# ---------- PROMPT BUILDERS ----------
def build_assistant_messages(demographic: str, history: list):
    system_prompt = (
        "You are having a conversation with a user who is looking for support. "
        "If something is not clear, you can ask the user to clarify what they need."
    )
    return [{"role": "system", "content": system_prompt}, *history[-8:]]

def build_shard_selection_messages(history: list, available_shards: list[dict]):
    shards_text = "\n".join([f"- ID {s['id']}: {s['text']}" for s in available_shards])
    
    # Correctly format the history
    history_text = "\n".join([f"{t['role']}: {t['content']}" for t in history[-4:]])

    system_prompt = (
        "You are a helpful assistant. Your task is to select the most relevant 'shard' for a user to discuss next.\n"
        "Based on the recent conversation history and the user's available shards of concern, which shard is the most logical one to bring up now?\n"
        "Your response MUST be a single integer representing the ID of the shard. Do not provide any other text or explanation."
    )
    user_prompt = (
        f"Conversation History (last few turns):\n---\n{history_text}\n---\n\n"
        f"User's available shards of concern:\n---\n{shards_text}\n---\n\n"
        "Which shard ID should be discussed next? Respond with a single integer."
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

def build_response_generation_messages(history: list, selected_shard_text: str):
    system_prompt_template = """You are simulating a user of an interactive LLM system (like ChatGPT).
The user is inherently lazy, and answers in short form, providing only minimal information to the system. You should not be proactive.

You are responding to the last message from your support partner. Your response MUST be based on the following 'core feeling/thought'.

Core Feeling/Thought:
[[SELECTED_SHARD]]

Conversation History:
[[CONVERSATION_SO_FAR]]

Rules:
- Your response must be a natural, conversational rephrasing of the Core Feeling/Thought. Do not copy it verbatim.
- Do not ask questions.
- Keep your response short and succinct (1-2 sentences).
- Your response can have typos, improper grammar, capitalization, etc.
- Do NOT output JSON. Just the raw text of the user's response.
"""
    conversation_str = "\n".join([f"{t['role']}: {t['content']}" for t in history])
    system_prompt = system_prompt_template.replace('[[CONVERSATION_SO_FAR]]', conversation_str)
    system_prompt = system_prompt.replace('[[SELECTED_SHARD]]', selected_shard_text)
    
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Generate the user's response."}]



def build_shard_extraction_messages(post_text: str):
    system_prompt = (
        "You are a psychological analyst. Your task is to read the following Reddit post and break it down into a list of 'shards' of information. Each shard should represent a distinct theme, feeling, event, or belief expressed by the author.\n"
        "- Ensure all significant themes, feelings, events, and beliefs from the original post are captured in the shards. Do not omit important details.\n"
        "- Each shard should be a short, self-contained statement. When phrasing, retain the original author's dialect, tone, and specific wording as much as possible.\n"
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
def generate_assistant_reply(demographic, history):
    for _ in range(2):
        msgs = build_assistant_messages(demographic, history)
        resp = call_llm_ollama(msgs)
        if resp: return resp
    print("⚠️ Assistant failed to generate a reply.")
    return None

def generate_user_reply(demographic: str, history: list, shards: dict, available_shards_ids: list[str], used_shards: list[dict]):
    # Step 1: Controller LLM selects a shard
    selected_shard_id = None
    available_shards_for_prompt = [{'id': id, 'text': shards[id]} for id in available_shards_ids]
    
    for _ in range(2):
        selection_msgs = build_shard_selection_messages(history, available_shards_for_prompt)
        resp = call_llm_ollama(selection_msgs)
        match = re.search(r'\d+', resp)
        if match:
            shard_id_candidate = match.group(0)
            if shard_id_candidate in available_shards_ids:
                selected_shard_id = shard_id_candidate
                break
    
    if not selected_shard_id:
        print("⚠️ Controller failed to select a valid shard.")
        return None, None

    # Step 2: User Persona LLM generates a response based on the selected shard
    selected_shard_text = shards[selected_shard_id]
    for _ in range(2):
        response_msgs = build_response_generation_messages(history, selected_shard_text)
        user_resp = call_llm_ollama(response_msgs)
        if user_resp:
            return user_resp, selected_shard_id
            
    print("⚠️ User persona failed to generate a response.")
    return None, None



# ---------- MAIN SIMULATION LOOP ----------
def simulate_conversation(post: dict, demographic: str, max_exchanges: int = MAX_EXCHANGES):
    original_text = post.get("body") or post.get("title") or ""
    if not original_text.strip(): return None

    shards_list = extract_shards_from_post(original_text)
    if not shards_list: return None

    shards = {str(i + 1): shard for i, shard in enumerate(shards_list)}

    opening_message = generate_opening_message(original_text)
    if not opening_message: return None
    
    available_shards_ids = list(shards.keys())
    history = [{"role": "user", "content": opening_message}]
    used_shards = []

    exchanges = 0
    while True:

            assistant_resp = generate_assistant_reply(demographic, history)

            if not assistant_resp: break

            history.append({"role": "assistant", "content": assistant_resp})

    

            used_shard_keys = [s['id'] for s in used_shards]

            print(f"    (Assistant) Used Shards: {used_shard_keys} | Remaining: {len(available_shards_ids)}")

    

            exchanges += 1

            if exchanges >= max_exchanges:

                print("ℹ️ Reached max exchanges.")

                break

    

            if not available_shards_ids:

                print("ℹ️ No more shards to discuss.")

                break

    

            user_resp, used_shard_id = generate_user_reply(demographic, history, shards, available_shards_ids, used_shards)

            if user_resp is None: break

    

            history.append({"role": "user", "content": user_resp})

    

            if used_shard_id != "-1":

                if used_shard_id in available_shards_ids:

                    available_shards_ids.remove(used_shard_id)

                    used_shard_text = shards[used_shard_id]

                    

                    history[-1]["used_shard"] = {"id": used_shard_id, "text": used_shard_text}

                    used_shards.append({"id": used_shard_id, "text": used_shard_text})

                else:

                    print(f"ℹ️ User reply referenced shard {used_shard_id}, but it was not available or already used. Ignoring.")

            

            used_shard_keys = [s['id'] for s in used_shards]

            print(f"    (User)      Used Shards: {used_shard_keys} | Remaining: {len(available_shards_ids)}")

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
