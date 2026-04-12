import pandas as pd
from transformers import AutoConfig

# Model IDs for OCR / Document Understanding Models
MODEL_NAMES = {
    "Openbmb": "openbmb/MiniCPM-V-2_6",
    "IBM": "ibm-granite/granite-4.0-3b-vision",
    "H2O": "h2oai/h2ovl-mississippi-2b",
    "GVLab": "OpenGVLab/InternVL2-1B",
    "Qwen": "Qwen/Qwen-VL",
    "Microsoft": "microsoft/phi-4",
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
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)

        # --- 1. Get Encoder Config (Vision / Image Encoder) ---
        encoder_config = getattr(config, 'encoder', None) or getattr(config, 'vision_config', None)

        # Some models (e.g. TrOCR/Donut) wrap encoder under 'encoder'
        # Others expose it directly at root level if purely vision-based
        if encoder_config is None and getattr(config, 'model_type', '') in ['vision-encoder-decoder']:
            encoder_config = config  # root IS the encoder config

        enc_layers = "N/A"
        enc_hidden_size = "N/A"
        enc_heads = "N/A"
        enc_patch_size = "N/A"
        enc_image_size = "N/A"

        if encoder_config:
            enc_layers = get_attr(encoder_config, ['num_hidden_layers', 'n_layer', 'encoder_layers'], "N/A")
            enc_hidden_size = get_attr(encoder_config, ['hidden_size', 'd_model'], "N/A")
            enc_heads = get_attr(encoder_config, ['num_attention_heads', 'encoder_attention_heads'], "N/A")
            enc_patch_size = get_attr(encoder_config, ['patch_size'], "N/A")
            enc_image_size = get_attr(encoder_config, ['image_size', 'input_size'], "N/A")
            print(f"    -> Encoder config found.")
        else:
            # Fallback: read directly from root config (e.g. GOT-OCR)
            enc_layers = get_attr(config, ['num_hidden_layers', 'n_layer'], "N/A")
            enc_hidden_size = get_attr(config, ['hidden_size', 'd_model'], "N/A")
            enc_heads = get_attr(config, ['num_attention_heads'], "N/A")
            enc_patch_size = get_attr(config, ['patch_size'], "N/A")
            enc_image_size = get_attr(config, ['image_size', 'input_size'], "N/A")
            print(f"    -> No separate encoder block; reading root config.")

        # --- 2. Get Decoder Config (Text Decoder) ---
        decoder_config = getattr(config, 'decoder', None) or getattr(config, 'text_config', None)

        dec_layers = "N/A (Encoder-Only)"
        dec_hidden_size = "N/A"
        dec_heads = "N/A"
        dec_vocab_size = "N/A"

        if decoder_config:
            dec_layers = get_attr(decoder_config, ['num_hidden_layers', 'n_layer', 'decoder_layers'], "N/A")
            dec_hidden_size = get_attr(decoder_config, ['hidden_size', 'd_model'], "N/A")
            dec_heads = get_attr(decoder_config, ['num_attention_heads', 'decoder_attention_heads'], "N/A")
            dec_vocab_size = get_attr(decoder_config, ['vocab_size'], "N/A")
            print(f"    -> Decoder config found.")
        else:
            # Some models have decoder params at root level
            dec_vocab_size = get_attr(config, ['vocab_size'], "N/A")
            print(f"    -> No separate decoder block found.")

        # --- 3. Capture Results ---
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": getattr(config, 'model_type', 'N/A').upper(),
            # Encoder (Vision / Image)
            "Encoder Layers": enc_layers,
            "Encoder Hidden Size": enc_hidden_size,
            "Encoder Attention Heads": enc_heads,
            "Encoder Patch Size": enc_patch_size,
            "Encoder Image Size": enc_image_size,
            # Decoder (Text)
            "Decoder Layers": dec_layers,
            "Decoder Hidden Size": dec_hidden_size,
            "Decoder Attention Heads": dec_heads,
            "Decoder Vocabulary Size": dec_vocab_size,
        })

    except Exception as e:
        print(f"❌ Failed to load configuration for {display_name}: {e}")
        data.append({
            "Model Name": display_name,
            "HF ID": hf_id,
            "Architecture Type": "Error",
            "Encoder Layers": "Failed",
            "Encoder Hidden Size": "Failed",
            "Encoder Attention Heads": "Failed",
            "Encoder Patch Size": "Failed",
            "Encoder Image Size": "Failed",
            "Decoder Layers": "Failed",
            "Decoder Hidden Size": "Failed",
            "Decoder Attention Heads": "Failed",
            "Decoder Vocabulary Size": "Failed",
        })

# --- Export to Excel ---
df = pd.DataFrame(data)

excel_file = "OCR_Model_Layer_Breakdown.xlsx"

try:
    df.to_excel(excel_file, index=False, sheet_name="OCR Architectures")
    print(f"\n✅ Successfully created Excel file: **{excel_file}**")
except Exception as e:
    print(f"\n❌ Error exporting to Excel: {e}")