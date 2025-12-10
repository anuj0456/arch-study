import pandas as pd
from transformers import AutoConfig

# Model IDs for Multimodal Models (Llama-4 and Gemma-3)
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
            # Handle booleans and other non-standard objects nicely
            if isinstance(value, bool):
                return 'Yes' if value else 'No'
            return value
    return default


# --- Data Extraction ---
data = []
for display_name, hf_id in MODEL_NAMES.items():
    print(f"Loading configuration for: {display_name}...")

    try:
        # Load the main composite configuration
        config = AutoConfig.from_pretrained(hf_id)

        # Determine the core text config object (handles multimodal models)
        text_config = getattr(config, 'text_config', config)

        # --- 1. Text (LLM) Modality ---
        text_layers = get_attr(text_config, ['num_hidden_layers', 'n_layer'], "N/A")
        text_hidden_size = get_attr(text_config, ['hidden_size'], "N/A")
        text_heads = get_attr(text_config, ['num_attention_heads'], "N/A")
        text_vocab = get_attr(text_config, ['vocab_size'], "N/A")

        # --- 2. Vision (Image Encoder) Modality ---
        # Vision models often use 'vision_config' or a similar attribute
        vision_config = getattr(config, 'vision_config', None)

        vision_layers = "N/A (No Vision)"
        vision_hidden_size = "N/A"
        vision_patch_size = "N/A"

        if vision_config:
            vision_layers = get_attr(vision_config, ['num_hidden_layers', 'n_layer'], "N/A")
            vision_hidden_size = get_attr(vision_config, ['hidden_size'], "N/A")
            vision_patch_size = get_attr(vision_config, ['patch_size'], "N/A")
            print(f"    -> Vision Modality found.")

        # --- 3. Mixture-of-Experts (MoE) Architecture ---
        # Llama-4 uses MoE, so we extract those specific parameters from the text config
        moe_experts = get_attr(text_config, ['num_local_experts', 'n_expert'], 0)
        moe_experts_per_tok = get_attr(text_config, ['num_experts_per_tok'], 0)

        # --- 4. Capture Results ---
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Base Architecture": getattr(config, 'model_type', 'N/A').upper(),
            # MoE Parameters
            "Is MoE?": 'Yes' if moe_experts > 1 else 'No',
            "Total Experts (Router)": moe_experts,
            "Experts Activated (per token)": moe_experts_per_tok,
            # Text Modality (LLM Decoder)
            "LLM Layers": text_layers,
            "LLM Hidden Size": text_hidden_size,
            "LLM Attention Heads": text_heads,
            "LLM Vocabulary Size": text_vocab,
            # Vision Modality (Image Encoder)
            "Vision Layers": vision_layers,
            "Vision Hidden Size": vision_hidden_size,
            "Vision Patch Size": vision_patch_size,
        })

    except Exception as e:
        print(f"❌ Failed to load configuration for {display_name}: {e}")
        # Append an error row for models that fail completely
        data.append({"Model Name": display_name, "HF ID": hf_id, "Base Architecture": "ERROR",
                     **{k: "Failed" for k in data[0].keys() if k not in ["Model Name", "HF ID", "Base Architecture"]}})

# --- Export to Excel ---
df = pd.DataFrame(data)

excel_file = "Multimodal_LLM_Architecture_Deep_Dive.xlsx"

try:
    df.to_excel(excel_file, index=False, sheet_name="Multimodal Layers")
    print(f"\n✅ Successfully created Excel file: **{excel_file}**")
    print("The Excel file includes: Text (LLM) Layers, Vision (Image Encoder) Layers, and MoE parameters.")
except Exception as e:
    print(f"\n❌ Error exporting to Excel: {e}")