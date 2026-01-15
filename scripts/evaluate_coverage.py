import json
import argparse
from pathlib import Path

def evaluate_coverage(conversation_file_path: Path):
    if not conversation_file_path.exists():
        print(f"Error: Conversation file not found at {conversation_file_path}")
        return

    with open(conversation_file_path, 'r') as f:
        data = json.load(f)

    all_detail_ids = set(data["shards"]["all"].keys())
    used_detail_ids = set(data["shards"]["used"])

    print(f"--- Coverage Evaluation for {conversation_file_path.name} ---")
    print(f"Total unique details identified: {len(all_detail_ids)}")
    print(f"Total unique details used in conversation: {len(used_detail_ids)}")

    if all_detail_ids == used_detail_ids:
        print("Result: SUCCESS - All identified details were used in the conversation.")
        return True
    else:
        missing_details = all_detail_ids - used_detail_ids
        print(f"Result: FAILURE - Not all identified details were used.")
        print(f"Missing details (IDs): {', '.join(sorted(list(missing_details)))}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Evaluates if all identified details from a Reddit post were used in a simulated conversation."
    )
    parser.add_argument(
        "conversation_file",
        type=str,
        help="Path to the conversation.json file to evaluate."
    )
    args = parser.parse_args()

    evaluate_coverage(Path(args.conversation_file))

if __name__ == "__main__":
    main()
