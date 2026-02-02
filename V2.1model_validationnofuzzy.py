import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

# --- CONFIGURATION ---
FILES = {
    "gold": "human_gold_standard.csv",
    "model": "devstral-2-123b.csv"
}

# Columns to compare
COMPARE_COLS = [
    "target_tense", 
    "correct_in_context", 
    "error_type", 
    "target_person_number",
    "is_missing"
]

def find_match(gold_row, model_df):
    """
    Finds a row in model_df that matches Student_ID, Context, and Verb exactly.
    """
    # 1. Identify the target verb (ensure it's a string and stripped of whitespace)
    target_verb = str(gold_row['verb_string']).strip()
    
    # 2. Filter model data for exact matches on ID, Context, and Verb
    # We use .astype(str) to ensure safe comparison even if data types differ slightly
    match = model_df[
        (model_df['Student_ID'] == gold_row['Student_ID']) & 
        (model_df['Full_Sentence_Context'] == gold_row['Full_Sentence_Context']) &
        (model_df['verb_string'].astype(str).str.strip() == target_verb)
    ]
    
    if not match.empty:
        # Return the first match found
        return match.iloc[0]
    else:
        return None

def main():
    print("Loading data...")
    df_gold = pd.read_csv(FILES["gold"])
    df_model = pd.read_csv(FILES["model"])
    
    comparison_data = []
    
    print(f"Validating {len(df_gold)} human-rated verbs (Exact Match Mode)...")
    
    for idx, gold_row in df_gold.iterrows():
        row_data = gold_row.to_dict()
        
        # Find Model Match
        match = find_match(gold_row, df_model)
        
        if match is not None:
            row_data['found_by_model'] = True
            for col in COMPARE_COLS:
                row_data[f"model_{col}"] = match.get(col)
        else:
            row_data['found_by_model'] = False
            for col in COMPARE_COLS:
                row_data[f"model_{col}"] = "NOT_FOUND"
                
        comparison_data.append(row_data)
        
    df_comp = pd.DataFrame(comparison_data)
    
    # --- CALCULATE METRICS ---
    print("\n" + "="*40)
    print(f"VALIDATION REPORT FOR: {FILES['model']}")
    print("="*40)
    
    # 1. Detection Rate (Recall)
    recall = df_comp['found_by_model'].mean()
    print(f"Detection Recall: {recall:.1%} (Did it find the verbs?)")
    
    # Filter only found verbs for accuracy metrics
    df_found = df_comp[df_comp['found_by_model'] == True].copy()
    
    if df_found.empty:
        print("No matches found! Check your column names or encoding.")
        return

    # 2. Tense Accuracy
    # Normalize strings to avoid "Preterite" vs "preterite" mismatches
    gold_tense = df_found['target_tense'].astype(str).str.lower().str.strip()
    model_tense = df_found['model_target_tense'].astype(str).str.lower().str.strip()
    
    tense_acc = accuracy_score(gold_tense, model_tense)
    print(f"Tense ID Accuracy: {tense_acc:.1%}")
    
    # 3. Error Detection Accuracy (Correct vs Incorrect)
    # Convert booleans to strings for comparison safety
    gold_corr = df_found['correct_in_context'].astype(str)
    model_corr = df_found['model_correct_in_context'].astype(str)
    
    err_acc = accuracy_score(gold_corr, model_corr)
    # Calculate Cohen's Kappa (Chance-corrected agreement)
    kappa = cohen_kappa_score(gold_corr, model_corr)
    
    print(f"Error Status Accuracy: {err_acc:.1%}")
    print(f"Cohen's Kappa: {kappa:.3f} ( >0.8 is excellent)")
    
    # 4. Save Disagreements
    disagreements = df_found[gold_tense != model_tense]
    if not disagreements.empty:
        print(f"\nFound {len(disagreements)} Tense Disagreements. Saving to 'disagreements.csv'...")
        disagreements.to_csv("disagreements.csv", index=False)
    else:
        print("\nPerfect Tense Agreement!")

if __name__ == "__main__":
    main()