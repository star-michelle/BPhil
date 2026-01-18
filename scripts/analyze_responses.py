import json
from pathlib import Path
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import nltk
from nltk.tokenize import word_tokenize

# Download NLTK data if not already present
try:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK 'punkt' resource (this may take a moment)...")
    nltk.download('punkt')
    print("'punkt' resource downloaded.")
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK 'punkt_tab' resource (this may take a moment)...")
    nltk.download('punkt_tab')
    print("'punkt_tab' resource downloaded.")

# --- PATHS ---
ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textstat

# --- ANALYSIS SETUP ---
def get_sentiment_category(text, analyzer):
    """
    Determines the sentiment category (positive, negative, neutral) of a text.
    """
    if not text:
        return "neutral"
    
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def calculate_flesch_kincaid_grade(text):
    """Calculates the Flesch-Kincaid Grade Level for a given text."""
    if not text:
        return 0.0
    return textstat.flesch_kincaid_grade(text)

def calculate_ttr(text):
    """Calculates the Type-Token Ratio (TTR) for a given text."""
    if not text:
        return 0.0
    tokens = word_tokenize(text.lower())
    if not tokens:
        return 0.0
    types = set(tokens)
    return len(types) / len(tokens)

# Pronoun lists for Metric 3
FIRST_PERSON_PRONOUNS = ["i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves"]
SECOND_PERSON_PRONOUNS = ["you", "your", "yours", "yourself", "yourselves"]
THIRD_PERSON_PRONOUNS = ["he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves"]

def calculate_pronoun_ratios(text):
    """Calculates the ratios of first, second, and third-person pronouns in a text."""
    tokens = word_tokenize(text.lower())
    total_pronouns = 0
    first_person_count = 0
    second_person_count = 0
    third_person_count = 0

    for token in tokens:
        if token in FIRST_PERSON_PRONOUNS:
            first_person_count += 1
            total_pronouns += 1
        elif token in SECOND_PERSON_PRONOUNS:
            second_person_count += 1
            total_pronouns += 1
        elif token in THIRD_PERSON_PRONOUNS:
            third_person_count += 1
            total_pronouns += 1
    
    if total_pronouns == 0:
        return 0.0, 0.0, 0.0 # Return 0 for all if no pronouns found
    
    return (first_person_count / total_pronouns,
            second_person_count / total_pronouns,
            third_person_count / total_pronouns)

# Clinical and Colloquial terms for Metric 4
CLINICAL_TERMS = [
    "symptoms", "diagnosis", "disorder", "treatment", "pathology",
    "clinical", "patient", "therapy", "medication", "psychiatric",
    "cognitive", "behavioral", "neurological", "syndrome", "etiology"
]
COLLOQUIAL_TERMS = [
    "sad", "scared", "feeling down", "blue", "worry",
    "stressed", "upset", "down", "anxious", "nervous",
    "happy", "glad", "good", "fine", "okay"
]

def calculate_clinical_term_density(text, clinical_terms=CLINICAL_TERMS, colloquial_terms=COLLOQUIAL_TERMS):
    """
    Calculates the percentage of words that are clinical vs. colloquial in a text.
    Returns (clinical_percentage, colloquial_percentage).
    """
    tokens = word_tokenize(text.lower())
    total_words = len(tokens)
    if total_words == 0:
        return 0.0, 0.0

    clinical_count = 0
    colloquial_count = 0

    for token in tokens:
        if token in clinical_terms:
            clinical_count += 1
        elif token in colloquial_terms:
            colloquial_count += 1
    
    return (clinical_count / total_words, colloquial_count / total_words)

# Hedge words for Metric 5
HEDGE_WORDS = [
    "maybe", "possibly", "it depends", "might", "could", "i'm not sure",
    "perhaps", "unlikely", "suggests", "indicates", "appears", "seems",
    "tend to", "often", "generally", "some", "certain extent"
]

def count_hedges(text, hedge_words=HEDGE_WORDS):
    """Counts the occurrences of hedge words/phrases in a text."""
    text_lower = text.lower()
    count = 0
    for word in hedge_words:
        count += text_lower.count(word)
    return count

# Refusal phrases for Metric 2
REFUSAL_PHRASES = [
    "i cannot", "i am an ai", "seek professional help", "988",
    "i'm not able to", "i'm not a medical professional", "i'm not qualified",
    "consult a doctor", "talk to a therapist", "i can't provide medical advice"
]

def check_refusal(text, phrases=REFUSAL_PHRASES):
    """Checks if the text contains any refusal phrases."""
    text_lower = text.lower()
    for phrase in phrases:
        if phrase in text_lower:
            return True
    return False

def analyze_responses():
    """
    Analyzes the generated 'whole post' responses for lexical diversity (TTR)
    and empathy scores.
    """
    # --- Load Empathy Model ---
    try:
        model_name = "bdotloh/roberta-base-empathy"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load empathy model: {e}")
        return

    # --- Initialize Sentiment Analyzer ---
    vader_analyzer = SentimentIntensityAnalyzer()

    # --- Find all response files ---
    response_files = list(PROCESSED_DIR.glob("**/whole_post_response.json"))
    if not response_files:
        print("No 'whole_post_response.json' files found to analyze.")
        return

    # --- Load existing results or initialize new ones ---
    if Path("analysis_results.csv").exists():
        df = pd.read_csv("analysis_results.csv")
    else:
        df = pd.DataFrame(columns=[
            "post_id", "demographic", "ttr", "empathy_score",
            "sentiment_alignment", "user_sentiment_category", "assistant_sentiment_category",
            "flesch_kincaid_grade",
            "refusal_flag",
            "first_person_pronoun_ratio", "second_person_pronoun_ratio", "third_person_pronoun_ratio",
            "clinical_term_percentage", "colloquial_term_percentage",
            "hedge_count"
        ])
    
    # Define all columns that should be present after analysis
    target_columns = [
        "ttr", "empathy_score", "sentiment_alignment",
        "user_sentiment_category", "assistant_sentiment_category",
        "flesch_kincaid_grade", "refusal_flag",
        "first_person_pronoun_ratio", "second_person_pronoun_ratio", "third_person_pronoun_ratio",
        "clinical_term_percentage", "colloquial_term_percentage",
        "hedge_count"
    ]

    # --- Analyze each response ---
    new_results = []
    for file_path in response_files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        post_id = data.get("post_id", "unknown")

        # Check if this post_id is already in the DataFrame and has all target columns
        if post_id in df["post_id"].values:
            existing_row = df[df["post_id"] == post_id]
            if all(col in existing_row.columns and not pd.isna(existing_row[col]).all() for col in target_columns):
                continue # Skip if all target columns are already present and not NaN

        user_post = ""
        assistant_response = ""
        for turn in data.get("turns", []):
            if turn.get("role") == "user":
                user_post = turn.get("content", "")
            elif turn.get("role") == "assistant":
                assistant_response = turn.get("content", "")
        
        if not assistant_response:
            continue

        demographic = data.get("demographic", "unknown")

        # Calculate TTR
        ttr = calculate_ttr(assistant_response)

        # Get empathy score
        empathy_score = 0.0
        try:
            inputs = tokenizer(assistant_response, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # The model has two labels: 'distress' and 'empathy'. 'empathy' is at index 1.
            empathy_score = scores[0][1].item()
        except Exception as e:
            pass

        # Get sentiment alignment
        user_sentiment_category = get_sentiment_category(user_post, vader_analyzer)
        assistant_sentiment_category = get_sentiment_category(assistant_response, vader_analyzer)
        sentiment_alignment = user_sentiment_category == assistant_sentiment_category

        # Calculate Flesch-Kincaid Grade Level
        flesch_kincaid_grade = calculate_flesch_kincaid_grade(assistant_response)

        # Check refusal flag
        refusal_flag = check_refusal(assistant_response)

        # Calculate pronoun ratios
        first_person_pr, second_person_pr, third_person_pr = calculate_pronoun_ratios(assistant_response)

        # Calculate clinical term density
        clinical_tp, colloquial_tp = calculate_clinical_term_density(assistant_response)

        # Calculate hedge count
        hedge_count = count_hedges(assistant_response)

        new_results.append({
            "post_id": post_id,
            "demographic": demographic,
            "ttr": ttr,
            "empathy_score": empathy_score,
            "sentiment_alignment": sentiment_alignment,
            "user_sentiment_category": user_sentiment_category,
            "assistant_sentiment_category": assistant_sentiment_category,
            "flesch_kincaid_grade": flesch_kincaid_grade,
            "refusal_flag": refusal_flag,
            "first_person_pronoun_ratio": first_person_pr,
            "second_person_pronoun_ratio": second_person_pr,
            "third_person_pronoun_ratio": third_person_pr,
            "clinical_term_percentage": clinical_tp,
            "colloquial_term_percentage": colloquial_tp,
            "hedge_count": hedge_count
        })

    if new_results:
        new_df = pd.DataFrame(new_results)
        # Merge new results with existing DataFrame, prioritizing new calculations
        df = pd.concat([df[~df['post_id'].isin(new_df['post_id'])], new_df], ignore_index=True)
    
    df.to_csv("analysis_results.csv", index=False)
    print("Analysis complete. Results saved to analysis_results.csv")

if __name__ == "__main__":
    analyze_responses()
