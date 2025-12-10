import pandas as pd
from transformers import AutoConfig

# Model IDs for Multimodal Models
MODEL_NAMES = {
    "Llama-4-Scout-Instruct": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
    "Llama-4-Scout": "meta-llama/Llama-4-Scout-17B-16E",
    "Llama-4-Maverick": "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
    "Gemma-3-27b": "google/gemma-3-27b-it",
}


# --- Utility Function for Safe Attribute Retrieval ---
def get_attr(config, names, default="N/A"):
    """Safely retrieves a parameter using a list of potential attribute names."""
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    return default


# --- Data Extraction ---
data = []
for display_name, hf_id in MODEL_NAMES.items():
    print(f"Loading configuration for: {display_name}...")

    try:
        # Load the main composite configuration
        config = AutoConfig.from_pretrained(hf_id)

        # --- 1. Get Text (LLM) Config ---
        # The text config holds the decoder layers (the LLM)
        text_config = getattr(config, 'text_config', config)  # Use main config as fallback

        text_layers = get_attr(text_config, ['num_hidden_layers', 'n_layer'], "N/A")
        text_hidden_size = get_attr(text_config, ['hidden_size'], "N/A")
        text_heads = get_attr(text_config, ['num_attention_heads'], "N/A")
        text_vocab = get_attr(text_config, ['vocab_size'], "N/A")

        # --- 2. Get Vision (Image Encoder) Config ---
        # Vision models often use 'vision_config' for the encoder layers (e.g., ViT)
        vision_config = getattr(config, 'vision_config', None)

        vision_layers = "N/A (Text-Only Modality)"
        vision_hidden_size = "N/A"
        vision_heads = "N/A"
        vision_patch_size = "N/A"

        if vision_config:
            vision_layers = get_attr(vision_config, ['num_hidden_layers', 'n_layer'], "N/A")
            vision_hidden_size = get_attr(vision_config, ['hidden_size'], "N/A")
            vision_heads = get_attr(vision_config, ['num_attention_heads'], "N/A")
            vision_patch_size = get_attr(vision_config, ['patch_size'], "N/A")
            print(f"    -> Multimodal config found with Vision component.")
        else:
            print("    -> Only Text Modality details captured.")

        # --- 3. Capture Results ---
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": getattr(config, 'model_type', 'N/A').upper(),
            # Text Modality (LLM Decoder)
            "LLM Layers": text_layers,
            "LLM Hidden Size": text_hidden_size,
            "LLM Attention Heads": text_heads,
            "LLM Vocabulary Size": text_vocab,
            # Vision Modality (Image Encoder)
            "Vision Layers": vision_layers,
            "Vision Hidden Size": vision_hidden_size,
            "Vision Attention Heads": vision_heads,
            "Vision Patch Size": vision_patch_size,
        })

    except Exception as e:
        print(f"❌ Failed to load configuration for {display_name}: {e}")
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": "Error",
            "LLM Layers": "Failed",
            "LLM Hidden Size": "Failed",
            "LLM Attention Heads": "Failed",
            "LLM Vocabulary Size": "Failed",
            "Vision Layers": "Failed",
            "Vision Hidden Size": "Failed",
            "Vision Attention Heads": "Failed",
            "Vision Patch Size": "Failed",
        })

# --- Export to Excel ---
df = pd.DataFrame(data)

excel_file = "Multimodal_LLM_Layer_Breakdown.xlsx"

try:
    df.to_excel(excel_file, index=False, sheet_name="Multimodal Architectures")
    print(f"\n✅ Successfully created Excel file: **{excel_file}**")
except Exception as e:
    print(f"\n❌ Error exporting to Excel: {e}")