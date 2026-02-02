#!/usr/bin/env python3
import pandas as pd
import unicodedata
from difflib import SequenceMatcher

# --- CONFIGURATION ---
FILES = {
    "anchor": "v2-gpt-oss120b.csv",           # Model 1 (The Anchor)
    "model_2": "v2-llama3-3-70b.csv",         # Model 2 (Validator 1)
    "model_3": "v2-mistral-small3-2-24b.csv"  # Model 3 (Validator 2)
}

FUZZY_THRESHOLD = 0.8

# Columns to COPY from secondary models for reference
COPY_COLS = [
    "verb_string", "target_form", "correct_in_context", "error_type", 
    "observed_tense", "target_tense", "observed_mood", "target_mood",
    "observed_person_number", "target_person_number", "is_missing", 
    "explanation", "infinitive", "subject_identification", 
    "is_reflexive", "lexical_class"
]

def normalize_text(text):
    """Aggressive normalization for fuzzy matching (strip accents, lowercase)."""
    if pd.isna(text): return ""
    text = str(text).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def fuzzy_match_score(s1, s2):
    return SequenceMatcher(None, normalize_text(s1), normalize_text(s2)).ratio()

def clean_val(val):
    """Standardize values for comparison (handle NaNs, strings)."""
    if pd.isna(val): return "MISSING"
    return str(val).lower().strip()

def find_match_in_model(anchor_row, model_df):
    """Finds the matching row in a secondary model using Context + Verb."""
    # Blocking: Filter by Student and Sentence Context first
    subset = model_df[
        (model_df['Student_ID'] == anchor_row['Student_ID']) & 
        (model_df['Full_Sentence_Context'] == anchor_row['Full_Sentence_Context'])
    ].copy()
    
    if subset.empty:
        return None
    
    best_score = 0
    best_row = None
    anchor_verb = anchor_row['verb_string']
    
    for idx, row in subset.iterrows():
        current_verb = str(row['verb_string']) if pd.notna(row['verb_string']) else ""
        score = fuzzy_match_score(str(anchor_verb), current_verb)
        
        if score > best_score:
            best_score = score
            best_row = row
            
    if best_score >= FUZZY_THRESHOLD:
        return best_row
    return None

def calculate_anchor_priority_consensus(row, col_name):
    # Get normalized values
    v_anchor = clean_val(row.get(col_name))
    v_m2 = clean_val(row.get(f"{col_name}_model_2"))
    v_m3 = clean_val(row.get(f"{col_name}_model_3"))
    
    # --- LOGIC STEP: Status & Decision ---
    if v_anchor == v_m2 == v_m3:
        status = 3  # Unanimous
        final_decision = row.get(col_name) # Trust Anchor
        
    elif v_anchor == v_m2 or v_anchor == v_m3:
        status = 2  # Anchor Supported
        final_decision = row.get(col_name) # Trust Anchor
        
    elif v_m2 == v_m3 and v_anchor != v_m2:
        status = 1  # Anchor Overruled by Majority
        final_decision = row.get(f"{col_name}_model_2") # Trust Majority 
        
    else:
        status = 0  # Total Chaos
        final_decision = row.get(col_name) # Default to Anchor
        
    return pd.Series([final_decision, status])

def main():
    print("Loading datasets...")
    try:
        df_anchor = pd.read_csv(FILES["anchor"])
        df_m2 = pd.read_csv(FILES["model_2"])
        df_m3 = pd.read_csv(FILES["model_3"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Aligning {len(df_anchor)} anchor verb events...")
    
    aligned_data = []
    
    for idx, anchor_row in df_anchor.iterrows():
        row_data = anchor_row.to_dict()
        
        # Match Model 2
        match_2 = find_match_in_model(anchor_row, df_m2)
        if match_2 is not None:
            for col in COPY_COLS:
                row_data[f"{col}_model_2"] = match_2.get(col)
        
        # Match Model 3
        match_3 = find_match_in_model(anchor_row, df_m3)
        if match_3 is not None:
            for col in COPY_COLS:
                row_data[f"{col}_model_3"] = match_3.get(col)
                
        aligned_data.append(row_data)
        
    master_df = pd.DataFrame(aligned_data)
    
    # --- RUN CONSENSUS ---
    print("Calculating Consensus Scores...")
    
    # 1. Main Binary Consensus (Correct vs Incorrect)
    res = master_df.apply(lambda r: calculate_anchor_priority_consensus(r, "correct_in_context"), axis=1)
    master_df['Final_Correctness'] = res[0]
    master_df['Correctness_Risk_Status'] = res[1]
    
    # 2. Detailed Feature Consensus (Target Tense ONLY)
    for feat in ["target_tense"]:
        res = master_df.apply(lambda r: calculate_anchor_priority_consensus(r, feat), axis=1)
        master_df[f'Final_{feat}'] = res[0]
        master_df[f'{feat}_Risk_Status'] = res[1]

    # Save
    output_filename = "master_anchor_priority.csv"
    master_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
    print("Done. Analysis complete.")
    print(f"Results saved to: {output_filename}")
    
    # --- REPORTING ---
    print("\n=== VALIDATION REPORT ===")
    print("Risk Status Legend: 3=Unanimous, 2=Anchor Supported, 1=Anchor Standalone (Majority wins), 0=Chaos")
    
    print("\nBinary Correctness Risk Levels:")
    print(master_df['Correctness_Risk_Status'].value_counts().sort_index())
    
    print("\nTarget Tense Risk Levels:")
    print(master_df['target_tense_Risk_Status'].value_counts().sort_index())

if __name__ == "__main__":
    main()
