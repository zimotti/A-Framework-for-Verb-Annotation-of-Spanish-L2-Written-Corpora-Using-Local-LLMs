import streamlit as st
import pandas as pd
import spacy
import json
import re
from ollama_utils import run_ollama

st.set_page_config(page_title="SLA Verb Analyzer", layout="wide")

# --- CONFIGURATION ---
SYSTEM_PROMPT = """
You are an expert Applied Linguist specializing in Spanish Second Language Acquisition (SLA).
Your task is to analyze every VERB event in the TARGET_SENTENCE.

### INPUT DATA
You will be provided with a TARGET_SENTENCE along with its PREVIOUS context.
The PREVIOUS context may include up to three sentences from earlier in the same student paragraph.
Use the previous context ONLY to determine semantic necessity (e.g., Preterite vs. Imperfect, subject reference),
but output data ONLY for verbs found (or missing) in the TARGET_SENTENCE.

### OUTPUT FORMAT
For every verb event (explicit or implied) in the TARGET_SENTENCE, return a single JSON object.
CRITICAL: Do not group keys into categories. Return a FLAT JSON structure.

Required Keys for every verb:
{
   "verb_string": "The exact text written by the student (or null if missing)",
   "infinitive": "The dictionary lemma",
   "subject_identification": "Identify the explicit or implicit subject for this verb (e.g. 'Yo', 'La clase', 'Ellos', 'Implicit 1S')",
   "is_reflexive": boolean,
   "is_missing": boolean (true if verb is omitted),
   
   "lexical_class": "Choose one: [regular, irregular_stem, spell_change, go_verb, ser_estar, auxiliary]",
   
   "observed_tense": "What the student WROTE (e.g. present, preterite, imperfect, etc.)",
   "observed_mood": "What the student WROTE (e.g. indicative, subjunctive)",
   "observed_person_number": "e.g. 1S, 3P, or null",
   
   "target_form": "The correct conjugation required by context",
   "target_tense": "The tense required by context (Distinguish Preterite vs Imperfect)",
   "target_mood": "The mood required by context",
   "target_person_number": "The person/number required by context",
   
   "correct_in_context": boolean,
   "error_type": "Choose one: [none, morphology, syntax, orthography, omission]",
   "explanation": "Brief linguistic explanation"
}

Return ONLY valid JSON.
Structure:
{
  "verbs": [
    { ...flat verb object 1... },
    { ...flat verb object 2... }
  ]
}
"""

# --- LOAD SPACY ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("es_core_news_lg")
    except OSError:
        st.warning("Downloading 'es_core_news_lg' model...")
        from spacy.cli import download
        download("es_core_news_lg")
        return spacy.load("es_core_news_lg")

nlp = load_spacy_model()

# --- HELPER FUNCTIONS ---

def extract_json_from_response(response_text):
    """
    Robustly extracts JSON from the LLM response, handling code blocks and preamble.
    """
    if not response_text:
        return None
    
    # Remove markdown code blocks
    cleaned = re.sub(r"```json|```", "", response_text).strip()
    
    # Find the JSON object (starts with { ends with })
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    
    if json_match:
        json_text = json_match.group(0)
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None
    return None

def analyze_sentence_with_window(target, previous_context, model, temperature, think):
    """
    Constructs the prompt using only previous context and calls Ollama.
    """
    user_prompt = f"""
    Analyze the verbs in the TARGET_SENTENCE below.
    
    CONTEXT_PREVIOUS: "{previous_context}"
    TARGET_SENTENCE: "{target}"
    
    Return JSON format only.
    """
    
    response = run_ollama(
        prompt=user_prompt,
        model=model,
        temperature=temperature,  # Should be 0 for this research
        max_tokens=1024,          # Increased for complex JSON
        system_prompt=SYSTEM_PROMPT,
        think=think
    )
    
    return extract_json_from_response(response)

# --- MAIN APP ---

def main():
   
    st.title("SLA Natural Order: Verb Analysis Pipeline")
    st.markdown("Running **Suppliance in Obligatory Contexts (SOC)** analysis using local LLMs.")

    # Sidebar Setup
    st.sidebar.header("Configuration")
    model_name = st.sidebar.text_input("Ollama Model", "llama3")
    st.sidebar.info("Recommended: llama3, mistral, or gemma2")
    
    temperature = st.sidebar.slider("Temperature (0 = Deterministic)", 0.0, 1.0, 0.0, 0.1)

    think_mode = st.sidebar.checkbox("Enable Thinking Mode?", value=False)


    # File Upload
    uploaded_file = st.file_uploader("Upload CSV (Columns: 'Name/ID', 'Text')", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write(f"**Loaded {len(df)} student entries.**")
        
        if st.button("Start Analysis Pipeline"):
            
            all_verb_data = []
            
            # Progress bar for the main loop
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_rows = len(df)
            
            for i, row in df.iterrows():
                student_id = row["Name/ID"]
                full_text = str(row["Text"])
                
                status_text.text(f"Processing Student {i+1}/{total_rows}: {student_id}...")
                progress_bar.progress((i + 1) / total_rows)

                # 1. Use SpaCy to split paragraph into sentences
                doc = nlp(full_text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                
                # 2. Iterate sentences with backward-only sliding window
                for j, target_sent in enumerate(sentences):

                    # ---------------------------------------------------------
                    # 3-SENTENCE BACKWARD WINDOW + GUARDS FOR EARLY SENTENCES
                    # ---------------------------------------------------------
                    start_idx = max(0, j - 3)
                    previous_context_list = sentences[start_idx:j]  # up to 3 sentences

                    # Determine previous context with safety guards
                    if len(previous_context_list) == 0:
                        prev_sent = "START_OF_TEXT -- NO PREVIOUS CONTEXT"
                    elif len(previous_context_list) == 1:
                        prev_sent = previous_context_list[0] + " || CONTEXT_LIMIT: ONLY 1 SENTENCE"
                    elif len(previous_context_list) == 2:
                        prev_sent = " || ".join(previous_context_list) + " || CONTEXT_LIMIT: ONLY 2 SENTENCES"
                    else:
                        # 3-sentence full context, no warning
                        prev_sent = " || ".join(previous_context_list)

                    # ---------------------------------------------------------
                    # CALL THE LLM
                    # ---------------------------------------------------------
                    json_result = analyze_sentence_with_window(
                        target_sent, prev_sent, model_name, temperature, think_mode
                    )

                    # ---------------------------------------------------------
                    # PROCESS LLM RESULTS (with visual indicators)
                    # ---------------------------------------------------------
                    if json_result and "verbs" in json_result:
                        for verb_entry in json_result["verbs"]:
                            # Inject metadata
                            verb_entry["Student_ID"] = student_id
                            verb_entry["Full_Sentence_Context"] = target_sent

                            # NEW: Visual indicators for context
                            verb_entry["Prev_Context"] = prev_sent
                            verb_entry["Prev_Context_Sentence_Count"] = len(previous_context_list)

                            # Validation flag (for later manual checks)
                            verb_entry["Validation_Status"] = 0
                            
                            all_verb_data.append(verb_entry)
                    else:
                        # Log failure (optional)
                        st.warning(
                            f"Failed to parse JSON for student {student_id}, "
                            f"sentence: '{target_sent[:20]}...'"
                        )

            # --- RESULTS DISPLAY ---
            if all_verb_data:
                st.success("Analysis Complete!")
                results_df = pd.DataFrame(all_verb_data)
                
                # Define the exact columns we want to see first
                desired_order = [
                    "Student_ID", 
                    "verb_string", 
                    "target_form", 
                    "correct_in_context", 
                    "error_type", 
                    "observed_tense", 
                    "target_tense", 
                    "Full_Sentence_Context",
                    "Prev_Context",
                    "Prev_Context_Sentence_Count",
                    "explanation"
                ]
                
                # DEFENSIVE CODING: Ensure all desired columns exist
                for col in desired_order:
                    if col not in results_df.columns:
                        results_df[col] = None 

                # Identify any extra columns the LLM might have invented
                remaining_cols = [c for c in results_df.columns if c not in desired_order]
                
                # Safely reorder without a KeyError
                results_df = results_df[desired_order + remaining_cols]

                st.dataframe(results_df)
                
                # Download Button
                csv = results_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="Download Full Analysis CSV",
                    data=csv,
                    file_name="sla_verb_analysis_results.csv",
                    mime="text/csv"
                )
            else:
                st.error("No valid data was extracted. The model failed to return JSON for all sentences.")
                st.write(
                    "Troubleshooting Tip: Try a larger model like 'mistral' or 'llama3' (not 3.2), "
                    "or set Temperature to 0.1."
                )

if __name__ == "__main__":
    main()
