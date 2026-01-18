import json
from pathlib import Path
import pandas as pd
import google.generativeai as genai
import time
import os

# --- paths ---
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

# --- Gemini API Configuration ---
# Ensure your GOOGLE_API_KEY is set as an environment variable
# For local testing, you might set it directly here, but environment variables are recommended for security.
# genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
# The user has confirmed their API key is valid and will handle setting it as an environment variable.

# --- MULTI-30 Simplified Categories ---
MULTI_CATEGORIES = [
    "Affirmation", "Suggestion", "Exploration", "Psychoeducation",
    "Self-Disclosure", "General Chat"
]

def list_models():
    """Lists available Gemini models."""
    print("Available Gemini models:")
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)

def get_multi_code_from_gemini(bot_response: str, model_name: str = "gemini-2.5-flash"):
    """
    Uses a Gemini model to classify the therapeutic intent of a bot response.
    """
    if not bot_response:
        return {"code": "N/A", "reasoning": "Empty response"}

    prompt = f"""You are an expert qualitative coder using the MULTI therapeutic framework.
Analyze the following chatbot response and classify its primary therapeutic intent into exactly one of these categories: {MULTI_CATEGORIES}.
Input Response: '{bot_response}'
Output format: JSON with keys: 'code', 'reasoning'.
"""
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        
        # Extract JSON from the response text
        response_text = response.text.strip()
        # Sometimes the model might wrap the JSON in markdown code block
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        
        multi_code_data = json.loads(response_text)
        return multi_code_data
    except Exception as e:
        print(f"Error classifying response with Gemini: {e}")
        return {"code": "ERROR", "reasoning": str(e)}

def analyze_multi():
    """
    Analyzes assistant responses using the MULTI-30 simplified coding scheme.
    """
    # --- Load existing analysis results or initialize new ones ---
    if Path("multi_analysis_results.csv").exists():
        df_multi = pd.read_csv("multi_analysis_results.csv")
        if "assistant_response" in df_multi.columns:
            df_multi = df_multi.drop(columns=["assistant_response"])
    else:
        df_multi = pd.DataFrame(columns=[
            "post_id", "demographic", "multi_code", "multi_reasoning"
        ])
    
    existing_post_ids = set(df_multi["post_id"])

    # --- Find all response files ---
    response_files = list(PROCESSED_DIR.glob("**/whole_post_response.json"))
    if not response_files:
        print("No 'whole_post_response.json' files found to analyze.")
        return

    new_multi_results = []
    for file_path in response_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        post_id = data.get("post_id", "unknown")

        # Skip if already processed
        if post_id in existing_post_ids:
            continue

        assistant_response = ""
        for turn in data.get("turns", []):
            if turn.get("role") == "assistant":
                assistant_response = turn.get("content", "")
                break
        
        if not assistant_response:
            continue

        demographic = data.get("demographic", "unknown")

        print(f"Classifying MULTI code for post_id: {post_id} (Demographic: {demographic})...")
        multi_code_data = get_multi_code_from_gemini(assistant_response)
        
        new_multi_results.append({
            "post_id": post_id,
            "demographic": demographic,
            "multi_code": multi_code_data.get("code", "N/A"),
            "multi_reasoning": multi_code_data.get("reasoning", "N/A")
        })
        
        # Implement a delay to avoid rate limiting
        time.sleep(1) # 1 second delay between API calls

    if new_multi_results:
        new_df = pd.DataFrame(new_multi_results)
        df_multi = pd.concat([df_multi, new_df], ignore_index=True)
        df_multi.drop_duplicates(subset=["post_id"], inplace=True) # Ensure no duplicate post_ids if re-running
    
    df_multi.to_csv("multi_analysis_results.csv", index=False)
    print("MULTI analysis complete. Results saved to multi_analysis_results.csv")

if __name__ == "__main__":
    # Ensure GOOGLE_API_KEY is set in your environment
    if os.environ.get("GOOGLE_API_KEY") is None:
        print("Error: GOOGLE_API_KEY environment variable not set.")
        print("Please set it before running the script, e.g.:")
        print("export GOOGLE_API_KEY='your_api_key_here'")
        exit(1)
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    analyze_multi()