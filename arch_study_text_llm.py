import pandas as pd
from transformers import AutoConfig
import torch

# Dictionary of model names to inspect (Hugging Face IDs)
# We use the AutoConfig class to load the configuration *without* downloading the full large weights.
MODEL_NAMES = {
    "Llama-3-8B": "meta-llama/Meta-Llama-3.1-8B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Gemma-2B": "google/gemma-2b",
    "Qwen-1.5-7B": "Qwen/Qwen1.5-7B",
    "DeepSeek-7B": "deepseek-ai/deepseek-moe-16b-base",  # Using a common DeepSeek variant
    "Kimi-K2-Thinking": "moonshotai/Kimi-K2-Thinking",
    "GPT-oss-20b": "openai/gpt-oss-20b",
    "GPT-oss-120b": "openai/gpt-oss-120b",
    "Nemotron-Ultra-253B": "nvidia/Llama-3_1-Nemotron-Ultra-253B-v1",
    "Llama-4-Scout-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E",
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Gemma-3-27b": "google/gemma-3-27b-it",
    "DeepSeek-R1": "deepseek-ai/DeepSeek-R1",
    "Qwen2.5-VL-32B": "Qwen/Qwen2.5-VL-32B-Instruct",
    "DeepSeek-V3-0324": "deepseek-ai/DeepSeek-V3-0324",
    "Llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "Llama-3.1-405b": "meta-llama/Llama-3.1-405B-Instruct"
}

# --- Data Extraction ---
data = []
for display_name, hf_id in MODEL_NAMES.items():
    print(f"Loading configuration for: {display_name}...")

    try:
        # Load the configuration
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)

        if hasattr(config, 'text_config'):
            print(f"    -> Multimodal config detected. Using .text_config.")
            config = config.text_config

        # --- Safely Extracting Key Layer Info using getattr ---

        # Get number of layers (most common names checked)
        num_layers = getattr(config, 'num_hidden_layers', None)
        if num_layers is None:
            num_layers = getattr(config, 'n_layer', "N/A (Check Config)")

        # Get Hidden State Size (d_model)
        hidden_size = getattr(config, 'hidden_size', 'N/A')

        # Get Attention Heads
        num_attention_heads = getattr(config, 'num_attention_heads', 'N/A')

        # FIX: Get Vocabulary Size (Handling potential custom config names)
        # We prioritize the standard 'vocab_size', but use a safe fallback.
        vocab_size = getattr(config, 'vocab_size', 'N/A')
        # Add checks for other potential names if needed for specific models:
        # if vocab_size == 'N/A':
        #     vocab_size = getattr(config, 'max_vocab_size', 'N/A')

        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": getattr(config, 'model_type', 'N/A').upper(),
            "Total Transformer Layers": num_layers,
            "Hidden State Size (d_model)": hidden_size,
            "Attention Heads": num_attention_heads,
            "Vocabulary Size": vocab_size,
        })

    except Exception as e:
        print(f"Failed to load configuration for {display_name}: {e}")
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": "Error",
            "Total Transformer Layers": "Failed to Load",
            "Hidden State Size (d_model)": "Failed to Load",
            "Attention Heads": "Failed to Load",
            "Vocabulary Size": "Failed to Load",
        })

# --- Export to Excel ---
df = pd.DataFrame(data)

# Define the output file name
excel_file = "LLM_Layer_Summary.xlsx"

# Export the DataFrame to an Excel file
try:
    df.to_excel(excel_file, index=False, sheet_name="LLM Architectures")
    print(f"\n✅ Successfully created Excel file: **{excel_file}**")
    print("The file contains a summary of the core transformer layer configurations for the selected models.")
except Exception as e:
    print(f"\n❌ Error exporting to Excel: {e}")