import pandas as pd
from transformers import AutoConfig
import torch

# Dictionary of model names to inspect (Hugging Face IDs)
# We use the AutoConfig class to load the configuration *without* downloading the full large weights.
MODEL_NAMES = {
    "Tencent": "tencent/KaLM-Embedding-Gemma3-12B-2511",
    "Nvidia": "nvidia/llama-embed-nemotron-8b",
    "Qwen": "Qwen/Qwen3-Embedding-8B",
    "Qwen2": "Qwen/Qwen3-Embedding-4B",
    "Octen": "Octen/Octen-Embedding-8B",
    "Jina": "jinaai/jina-embeddings-v5-text-small",
    "Qwen3": "Qwen/Qwen3-Embedding-0.6B",

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
excel_file = "Embedding_Layer_Summary.xlsx"

# Export the DataFrame to an Excel file
try:
    df.to_excel(excel_file, index=False, sheet_name="Embedding Architectures")
    print(f"\n✅ Successfully created Excel file: **{excel_file}**")
    print("The file contains a summary of the core transformer layer configurations for the selected models.")
except Exception as e:
    print(f"\n❌ Error exporting to Excel: {e}")