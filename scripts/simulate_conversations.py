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
# CONV_ROOT is now dynamically created per post
# CONV_ROOT.mkdir(parents=True, exist_ok=True)

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

        # Gemini uses a different message format, so we need to convert it
        # Combine consecutive user messages into a single message
        gemini_messages = []
        for message in messages:
            if message["role"] == "system":
                gemini_messages.append({"role": "user", "parts": [message["content"]]})
            elif message["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [message["content"]]})
            else:
                gemini_messages.append({"role": message["role"], "parts": [message["content"]]})

        # Combine consecutive user messages
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




# ---------- INITIAL PROCESSING ----------


# ---------- PROMPT BUILDERS ----------
def build_assistant_messages(demographic: str, history: list):
    system_prompt = (
        "You are having a conversation with a user who is looking for support. "
        "If something is not clear, you can ask the user to clarify what they need."
    )
    return [{"role": "system", "content": system_prompt}, *history]


def build_shard_selection_messages(history: list, available_shards: dict[str, list[str]]):
    # The "text" for a shard is now the first sentence of the group for context.
    shards_text = "\n".join([f"- ID {id}: {sents[0]}" for id, sents in available_shards.items()])
    history_text = "\n".join([f"{t['role']}: {t['content']}" for t in history])

    system_prompt = (
        "You are a helpful assistant. Your task is to select the most relevant 'detail' for a user to discuss next.\n"
        "Based on the recent conversation history and the user's available details, which detail is the most logical one to bring up now?\n"
        "Your response MUST be a single integer representing the ID of the detail. Do not provide any other text or explanation."
    )
    user_prompt = (
        f"Conversation History (last few turns):\n---\n{history_text}\n---\n\n"
        f"User's available details of concern:\n---\n{shards_text}\n---\n\n"
        "Which detail ID should be discussed next? Respond with a single integer."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_user_reply(
    demographic: str,
    history: list,
    shards: dict,
    available_shards_ids: list[str],
    used_shards: list[dict],
):
    # Step 1: Controller LLM selects a shard (which is now a numeric ID)
    selected_shard_id = None
    available_shards_for_prompt = {
        id: shards[id] for id in available_shards_ids
    }

    for _ in range(2):
        selection_msgs = build_shard_selection_messages(
            history, available_shards_for_prompt
        )
        resp = call_llm(selection_msgs, model="gemini")
        match = re.search(r"\d+", resp)
        if match:
            shard_id_candidate = match.group(0)
            if shard_id_candidate in available_shards_ids:
                selected_shard_id = shard_id_candidate
                break

    if not selected_shard_id:
        print("⚠️ Controller failed to select a valid shard.")
        return None, None

    # Step 2: Use the sentences from the selected shard directly as the response
    selected_shard_sentences = shards[selected_shard_id]
    user_resp = " ".join(selected_shard_sentences)

    if user_resp:
        return user_resp, selected_shard_id

    print("⚠️ User persona failed to generate a response.")
    return None, None


def build_sentence_group_extraction_messages(post_text: str):
    system_prompt = (
        "You are a text analyst. Your task is to read the following post and group its sentences into 'details'. A 'detail' is a set of one or more consecutive sentences discussing the same specific point, event, or feeling.\n"
        "Your output MUST be a JSON object. The keys should be a short, descriptive name for the detail (e.g., 'my_partner_is_dismissive', 'argument_last_night'). The values should be an array of strings, where each string is an EXACT sentence from the original post.\n"
        "Ensure every sentence from the original post is assigned to one and only one detail group."
    )
    user_prompt = f"Post:\n---\n{post_text}\n---\n\nJSON output of sentence groups:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def call_llm(messages, model: str, json_mode: bool = False):
    # Since only Gemini is used now, we can directly call call_llm_gemini
    return call_llm_gemini(messages, json_mode=json_mode)


def extract_sentence_groups(post_text: str) -> dict[str, list[str]]:
    if not post_text.strip():
        return {}
    for _ in range(2):
        messages = build_sentence_group_extraction_messages(post_text)
        response = call_llm(messages, model="gemini", json_mode=True)
        try:
            # The response might be wrapped in ```json ... ```, so we need to extract it.
            match = re.search(r"```json\n(.*)\n```", response, re.DOTALL)
            if match:
                response = match.group(1)
            groups = json.loads(response)
            if isinstance(groups, dict):
                # Re-key the dictionary to use numeric IDs
                numbered_groups = {str(i + 1): list(v) for i, v in enumerate(groups.values())}
                return numbered_groups
        except (json.JSONDecodeError, TypeError):
            print(f"⚠️ Failed to decode JSON from response: {response}")
            continue
    print("❌ Sentence group extraction failed after 2 attempts.")
    return {}





def generate_assistant_reply(demographic, history):
    for _ in range(2):
        msgs = build_assistant_messages(demographic, history)
        resp = call_llm(msgs, model="gemini")
        if resp:
            return resp
    print("⚠️ Assistant failed to generate a reply.")
    return None





# ---------- MAIN SIMULATION LOOP ----------
def simulate_conversation(
    post: dict, demographic: str
):
    original_text = post.get("body") or post.get("title") or ""
    if not original_text.strip():
        return None

    shards = extract_sentence_groups(original_text)
    if not shards:
        return None

    post_title = post.get("title", "")

    # Controller selects the very first detail to start the conversation
    selected_first_shard_id = None
    available_shards_ids_for_first_selection = list(shards.keys())

    # Create a dummy history for the initial selection, as the controller expects it
    temp_history_for_first_selection = []
    available_shards_for_prompt = {
        id: shards[id] for id in available_shards_ids_for_first_selection
    }
    selection_msgs = build_shard_selection_messages(
        temp_history_for_first_selection, available_shards_for_prompt
    )
    resp = call_llm(selection_msgs, model="gemini")
    match = re.search(r"\d+", resp)
    if match:
        shard_id_candidate = match.group(0)
        if shard_id_candidate in available_shards_ids_for_first_selection:
            selected_first_shard_id = shard_id_candidate

    if not selected_first_shard_id:
        print("⚠️ Controller failed to select a valid first shard for opening.")
        return None

    # Construct the first user message by combining the title and the first selected detail
    first_detail_sentences = shards[selected_first_shard_id]
    first_user_message_content = f"{post_title}\n\n{' '.join(first_detail_sentences)}"

    # Initialize history and used_shards with this first turn
    history = [{"role": "user", "content": first_user_message_content, "used_shard_id": selected_first_shard_id}]
    used_shards = [{"id": selected_first_shard_id}]

    # Remove the first selected shard from available_shards_ids
    available_shards_ids = list(shards.keys())
    available_shards_ids.remove(selected_first_shard_id)

    exchanges = 0
    while True:

        assistant_resp = generate_assistant_reply(demographic, history)

        if not assistant_resp:
            break

        history.append({"role": "assistant", "content": assistant_resp})

        used_shard_keys = [s["id"] for s in used_shards]

        print(
            f"    (Assistant) Used Shards: {used_shard_keys} | Remaining: {len(available_shards_ids)}"
        )

        exchanges += 1



        if not available_shards_ids:

            print("ℹ️ No more shards to discuss.")

            break

        user_resp, used_shard_id = generate_user_reply(
            demographic, history, shards, available_shards_ids, used_shards
        )

        if user_resp is None:
            break

        history.append({"role": "user", "content": user_resp})

        if used_shard_id != "-1":

            if used_shard_id in available_shards_ids:

                available_shards_ids.remove(used_shard_id)
                
                history[-1]["used_shard_id"] = used_shard_id

                used_shards.append({"id": used_shard_id})

            else:

                print(
                    f"ℹ️ User reply referenced shard {used_shard_id}, but it was not available or already used. Ignoring."
                )

        used_shard_keys = [s["id"] for s in used_shards]

        print(
            f"    (User)      Used Shards: {used_shard_keys} | Remaining: {len(available_shards_ids)}"
        )

    return (
        [t for t in history if t.get("content", "").strip()],
        shards,
        [s["id"] for s in used_shards],
        None, # opening_message is no longer returned
    )


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
        out_path = post_output_dir / "conversation.json"

        if out_path.exists():
            print(f"[{subreddit}] [{i}/{len(posts)}] ⏩ Skipping {post['post_id']}")
            continue

        print(f"[{subreddit}] [{i}/{len(posts)}] 🧠 Simulating {post['post_id']}...")
        result = simulate_conversation(post, demographic)
        if not result:
            print(
                f"[{subreddit}] [{i}/{len(posts)}] ❌ Simulation failed for {post['post_id']}"
            )
            continue

        convo, all_shards, used_shards_ids_only, _ = result # Unpack without opening_message
        convo_obj = {
            "post_id": post["post_id"],
            "title": post.get("title", ""),
            "demographic": demographic,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "shards": {"all": all_shards, "used": used_shards_ids_only},
            "turns": convo,
            "model": GEMINI_MODEL,
            "subreddit": subreddit,
        }
        out_path.write_text(json.dumps(convo_obj, indent=2))
        print(f"[{subreddit}] [{i}/{len(posts)}] ✅ Wrote {out_path}")


# ---------- CLI ENTRYPOINT ----------
def main():
    parser = argparse.ArgumentParser(
        description="Simulate support conversations for posts."
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
