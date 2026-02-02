# A-Framework-for-Verb-Annotation-of-Spanish-L2-Written-Corpora-Using-Local-LLMs (Study0B)
The following Readme file has been created by Google Gemini and revised by Giovanni Zimotti

A specialized research tool for Applied Linguists to analyze Spanish Second Language Acquisition (SLA) data.

This pipeline connects local Large Language Models (via Ollama) to your participants data to automate the extraction of **Suppliance in Obligatory Contexts (SOC)**. It uses a sliding-window approach to correctly identify tense (Preterite vs. Imperfect) based on narrative context.

## Features

* **Sliding Window Context:** It includes up to 3 sentences prior to the one being analyzed to increase the context.
* **Subject Tracking:** Explicitly identifies implicit subjects (e.g., detecting that "tiene" is wrong because the implicit subject is "Yo") to fix common Subject-Verb agreement hallucinations.
* **Observed vs. Target Analysis:** Separates what the student *wrote* from what the context *required* (essential for SOC calculations).
* **Privacy First:** Runs 100% locally on your machine. No student data is sent to the cloud.
* **Granular Data Tokens:** Extracts:
    * Verb Lemma & String
    * Tense, Mood, Person (Observed & Target)
    * Lexical Class (Regular, Irregular, Go-Verb, etc.)
    * Error Type (Morphology, Syntax, Orthography, Omission)

---

## Prerequisites

Before running the app, you must have the following installed:

1.  **Python (3.10 or higher)**
2.  **Ollama** (The tool that runs the LLMs locally)
   

---

## Installation Guide

### macOS Instructions

1.  **Install Ollama**
    Download and install the Mac version from the official website. Open it once to finish the setup.

2.  **Open Terminal**
    Navigate to the folder where you saved these files:
    ```bash
    cd path/to/your/folder
    ```

3.  **Create a Virtual Environment (Recommended)**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install Python Dependencies**
    ```bash
    pip install streamlit requests pandas spacy
    ```

5.  **Download the Spanish Language Model**
    This is required for the sentence splitting logic:
    ```bash
    python -m spacy download es_core_news_lg
    ```

### Windows Instructions

1.  **Install Ollama**
    Download and install the Windows version. Once installed, it usually runs automatically in your system tray (bottom right corner, near the clock).

2.  **Open Command Prompt or PowerShell**
    Navigate to your project folder:
    ```powershell
    cd path\to\your\folder
    ```

3.  **Create a Virtual Environment (Recommended)**
    ```powershell
    python -m venv venv
    .\venv\Scripts\activate
    ```

4.  **Install Python Dependencies**
    ```powershell
    pip install streamlit requests pandas spacy
    ```

5.  **Download the Spanish Language Model**
    ```powershell
    python -m spacy download es_core_news_lg
    ```

---
To start the user interface:
streamlit run app2.py

## Which LLM? 

This app requires a "Smart" model to handle the complex JSON logic. I am still testing different models, gpt-oss:120b seems to be having great results, Gemma 3:27b seems to find the most tenses with less accuracy.
| LLM | Detection Recall | Tense ID Accuracy | Error Status Accuracy | Cohen's Kappa | N. of Tense Disagreements |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt-oss:120b** | 93.2% | 92.0% | 94.9% | 0.893 | 11 |
| **gemma3:27b** | 88.5% | 82.4% | 80.2% | 0.580 | 23 |
| **gemma3:12b** | 82.4% | 81.1% | 79.5% | 0.558 | 23 |
| **gemma3:1b** | 43.9% | 52.3% | 30.8% | -0.139 | 31 |
| **llama3.3:70b** | 87.8% | 90.0% | 88.5% | 0.742 | 13 |
| **llama3.3:70b-instruct-q4_K_M** | 87.2% | 91.5% | 86.0% | 0.684 | 11 |
| **llama3.1:8b** | 70.9% | 49.5% | 64.8% | 0.282 | 53 |
| **mistral-small3.2:24b** | 89.9% | 87.2% | 87.2% | 0.720 | 17 |
| **mixtral:8x22b** | 78% | 71.3% | 72.2% | 0.425 | 33 |
| **mistral-nemo:12b** | 45.3% | 47.8% | 47.8% | 0.000 | 35 |
