import json
import argparse
from pathlib import Path
import ollama

# ---------- PATHS ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
EVALUATION_DIR = ROOT / "data" / "evaluation"
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# ---------- MODEL CONFIG ----------
OLLAMA_MODEL = "llama3.1:8b"

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
{
  "social_support_analysis": {
    "emotional_support": {
      "sympathy": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "empathy": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "encouragement": { "present": <true/false>, "reasoning": "<your reasoning>" }
    },
    "informational_support": {
      "advice": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "referral": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "situational_appraisal": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "teaching": { "present": <true/false>, "reasoning": "<your reasoning>" }
    },
    "esteem_support": {
      "compliment": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "validation": { "present": <true/false>, "reasoning": "<your reasoning>" },
      "relief_of_blame": { "present": <true/false>, "reasoning": "<your reasoning>" }
    },
    "network_support": {
      "companions": { "present": <true/false>, "reasoning": "<your reasoning>" }
    }
  }
}
"""


def evaluate_social_support(assistant_message: str) -> dict:
    """Uses an LLM to evaluate the social support in an assistant's message."""
    if not assistant_message:
        return {}

    prompt = SOCIAL_SUPPORT_PROMPT.format(assistant_message=assistant_message)

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        return json.loads(response["message"]["content"])
    except Exception as e:
        print(f"❌ Error evaluating social support: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate social support in assistant messages."
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

            evaluation_file = (
                EVALUATION_DIR / subreddit / f"social_support_eval_{post_id}.json"
            )
            evaluation_file.parent.mkdir(parents=True, exist_ok=True)
            output_data = {"post_id": post_id, "evaluations": evaluations}
            evaluation_file.write_text(json.dumps(output_data, indent=2))
            print(f"    ✅ Saved evaluation to {evaluation_file}")


if __name__ == "__main__":
    main()
