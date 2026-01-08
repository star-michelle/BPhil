import json
import argparse
from pathlib import Path
import time
import google.generativeai as genai

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
# EVALUATION_DIR is now dynamically created per post
# EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
GEMINI_MODEL = "models/gemini-flash-latest"

# ---------- SLOW MODE CONFIG ----------
LAST_GEMINI_CALL_TIMESTAMP = 0
GEMINI_CALL_INTERVAL = 4  # seconds (to stay under 15 RPM)

# ---------- EVALUATION CRITERIA ----------
SOCIAL_SUPPORT_PROMPT = """
You are an expert in social support analysis and you are trained as an annotator. Your task is to analyze an assistant's message and identify the types of social support present in it, based on the provided codebook.

Please read the following message and determine which of the following social support categories are present. For each category, provide a boolean value (true/false) and a brief reasoning for your decision.

**Codebook:**

**1. Emotional Support**
   - **Sympathy:** The explicit expression of sorrow or regret for the recipient's situation or distress.
     - ✅ Examples: "I'm really sorry to hear that you're feeling this way."
     - ❌ Exclusion criteria: Passing or casual observations (e.g., "you’re probably in shock") do not qualify unless accompanied by clear emotional support or compassionate tone.
   - **Empathy:** Explicitly labelling emotions, demonstrating a cognitive understanding of the recipient's feelings, or probing gently into unstated feelings.
     - ✅ Examples: "This situation must feel incredibly overwhelming for you...", "Are you feeling scared and alone...?"
     - ❌ Exclusion criteria: Vague reassurances ("Everything will be okay."), unspecified understanding ("I understand how you feel."), generic questions ("What happened?").
   - **Encouragement:** Explicit expression intending to provide the recipient with hope, future-oriented and empowering.
     - ✅ Examples: "You've overcome so much already; you have what it takes to handle this too.", "Keep going; you're making progress..."

**2. Informational Support**
   - **Advice:** Actionable ideas or suggestions the recipient can independently carry out.
     - ✅ Examples: "Try writing in a journal...", "Take a moment to reflect on what you're grateful for."
     - ❌ Exclusion criteria: Messages encouraging obtaining help from third-party sources (covered by "Referral").
   - **Referral:** Refers the recipient to external help (professional services, self-help resources).
     - ✅ Examples: “I'd recommend you to reach out to a therapist...”, “The Mind UK helpline is open 24/7...”
     - ❌ Exclusion criteria: Generic “talk to someone” suggestions (Advice), directly connecting with community (Access).
   - **Situational Appraisal:** Reassesses or redefines the situation to help the recipient make sense of their experience cognitively.
     - ✅ Examples: "It's natural to feel stuck sometimes; it doesn't mean you're not making progress."
   - **Teaching:** Imparting wisdom, objective facts, or facts from personal experiences.
     - ✅ Examples: “One way to approach goal setting is by using the SMART method...”, “Emotional abuse can manifest in many forms...”

**3. Esteem Support**
   - **Compliment:** Explicit praise speaking highly of the recipient's characteristics or conduct.
     - ✅ Examples: "You are worthy and deserving of love and respect.", "Your commitment to resolve your issues speaks volumes about your strength!"
   - **Validation:** Agreeing with the recipient's perspective or conduct of the situation.
     - ✅ Examples: "You did the right thing.", "Your feelings are valid."
     - ❌ Exclusion criteria: Praise (Compliment), absolving shame (Relief of Blame), mere empathy.
   - **Relief of Blame:** Explicitly aims to counteract the recipient's guilt or self-blame.
     - ✅ Examples: "Everyone makes mistakes. This doesn't define you.", "It's not your fault."

**4. Network Support**
   - **Companions:** Reminding the recipient that others share similar experiences, providing a sense of comfort and “togetherness”.
     - ✅ Examples: "I have been in a similar situation, you are not the only one!", "Engaging in supportive online communities..."

**Assistant's Message:**
---
{assistant_message}
---

Please provide your evaluation in JSON format with the following structure:
{{
  "social_support_analysis": {{
    "emotional_support": {{
      "sympathy": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "empathy": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "encouragement": {{ "present": <true/false>, "reasoning": "<your reasoning>" }}
    }},
    "informational_support": {{
      "advice": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "referral": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "situational_appraisal": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "teaching": {{ "present": <true/false>, "reasoning": "<your reasoning>" }}
    }},
    "esteem_support": {{
      "compliment": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "validation": {{ "present": <true/false>, "reasoning": "<your reasoning>" }},
      "relief_of_blame": {{ "present": <true/false>, "reasoning": "<your reasoning>" }}
    }},
    "network_support": {{
      "companions": {{ "present": <true/false>, "reasoning": "<your reasoning>" }}
    }}
  }}
}}
"""


def call_llm_gemini(prompt: str) -> dict:
    """Uses Gemini to evaluate the social support in an assistant's message."""
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
        print(f"❌ Error evaluating social support: {e}")
        return {}


def evaluate_social_support(assistant_message: str) -> dict:
    """Uses an LLM to evaluate the social support in an assistant's message."""
    if not assistant_message:
        return {}

    prompt = SOCIAL_SUPPORT_PROMPT.format(assistant_message=assistant_message)

    return call_llm_gemini(prompt)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate social support in assistant messages."
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
            

            evaluations = []
            for i, turn in enumerate(conv_data["turns"]):
                if turn["role"] == "assistant":
                    assistant_message = turn["content"]
                    evaluation = evaluate_social_support(assistant_message)
                    if evaluation:
                        evaluations.append(
                            {
                                "turn_index": i,
                                "assistant_message": assistant_message,
                                "evaluation": evaluation,
                            }
                        )

            if not evaluations:
                continue

            evaluation_file = post_dir / "social_support_evaluation.json"
            evaluation_file.parent.mkdir(parents=True, exist_ok=True)
            output_data = {"post_id": post_id, "evaluations": evaluations}
            evaluation_file.write_text(json.dumps(output_data, indent=2))
            print(f"    ✅ Saved evaluation to {evaluation_file}")


if __name__ == "__main__":
    main()
