# ollama_utils.py

import requests

def run_ollama(prompt,
               model="llama3.2",
               temperature=0.0,
               max_tokens=512,
               system_prompt=None,
               stream=False,
               think=False):
    """
    Sends a prompt to the local Ollama server and returns the generated text.

    Args:
        prompt (str): The user prompt.
        model (str): Model name (e.g. 'llama2', 'llama2:13b').
        temperature (float): Sampling temperature.
        max_tokens (int): Maximum tokens for the response.
        system_prompt (str): Optional system role content.
        stream (bool): Whether to request streaming response (usually leave False).
        think (bool): Enable/disable thinking for models that support it.

    Returns:
        str: The generated text from Ollama.
    """
    url = "http://127.0.0.1:11435/api/generate"
    payload = {
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "think": think,
    }

    # If you want to include a system prompt:
    if system_prompt:
        payload["system"] = system_prompt

    # Send request to Ollama server
    response = requests.post(url, json=payload, timeout=None)
    response.raise_for_status()

    # Ollama returns JSON with a "response" field.
    data = response.json()

    # If thinking is enabled Ollama returns:
    # { "response": "...", "thinking": "..." }
    if think and "thinking" in data:
        return data.get("response", "")  # we hide thinking for now
    else:
        return data.get("response", "")
