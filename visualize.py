import graphviz

from arch_study_deep_dive import moe_experts


def visualize_multimodal_moe(model_name, num_layers, is_moe, num_experts=0, has_vision=False):
    """
    Generates a structural diagram of the LLM, accounting for MoE and Vision components.
    """
    print(f"\nGenerating detailed graph for {model_name}...")

    # --- Graph Setup ---
    dot = graphviz.Digraph(
        comment=f'{model_name} Architecture',
        graph_attr={'rankdir': 'TB', 'splines': 'ortho', 'bgcolor': 'transparent'},
        node_attr={'shape': 'box', 'style': 'filled', 'fontname': 'Helvetica'}
    )

    # --- 1. Inputs ---
    dot.node('Text_Input', 'Text Input\n(Embeddings)', fillcolor='#AEC6E3')
    current_node = 'Text_Input'

    if has_vision:
        # --- 2. Vision Encoder Block (Separate Modality) ---
        dot.node('Vision_Input', 'Image Input', fillcolor='#FFA07A')

        # Group the vision encoder layers
        with dot.subgraph(name='cluster_V') as v:
            v.attr(rank='same', label='Vision Encoder (e.g., ViT)', style='dashed')
            v.node('V_ENC', 'Vision Encoder Block\n(Extracts Image Features)')

        dot.edge('Vision_Input', 'V_ENC')

        # Fusion Node: Where Vision features meet Text stream
        dot.node('Fusion', 'Multimodal Fusion Layer', fillcolor='#F08080')
        dot.edge('V_ENC', 'Fusion', label='Projected Features')
        dot.edge('Text_Input', 'Fusion', label='Initial Text Embeddings')

        current_node = 'Fusion'

    # --- 3. Core Layer Stack ---
    for i in range(num_layers):
        layer_id = f'L{i}'

        if is_moe and moe_experts > 1:
            # --- MoE Block Visualization ---
            block_color = '#FFD700'
            layer_label = f'MoE Layer {i}\n(Router + Experts)'

            # Create a subgraph cluster to represent the parallel experts
            with dot.subgraph(name=f'cluster_E{i}') as e:
                e.attr(label='MoE Experts', style='filled', color='#FFD70040', rankdir='LR')
                e.node(f'E{i}_Router', 'MoE Router', shape='diamond', fillcolor='#FFD700')
                e.node(f'E{i}_FFN1', 'Expert FFN 1', shape='box')
                e.node(f'E{i}_FFN2', 'Expert FFN 2', shape='box')
                e.node(f'E{i}_FFN_N', f'... Expert N ({num_experts} Total)', shape='box')

                # Internal connections within the MoE layer
                e.edge(f'E{i}_Router', f'E{i}_FFN1')
                e.edge(f'E{i}_Router', f'E{i}_FFN2')
                e.edge(f'E{i}_Router', f'E{i}_FFN_N')

            # Connect the previous layer/fusion node to the new MoE layer
            dot.node(layer_id, f'Attention + {layer_label}', fillcolor=block_color)
            dot.edge(current_node, layer_id)

            # Add a connection from the attention output to the MoE subgraph for realism
            dot.edge(layer_id, f'E{i}_Router', style='dotted', label='Attention Output')

        else:
            # --- Standard Transformer Block ---
            block_color = '#B0E0E6'
            layer_label = f'Transformer Block {i}\n(Attention + FFN)'
            dot.node(layer_id, layer_label, fillcolor=block_color)
            dot.edge(current_node, layer_id)

        current_node = layer_id

    # --- 4. Output ---
    dot.node('Output', 'Output Logits\n(Classification/Generation)', fillcolor='#AEC6E3')
    dot.edge(current_node, 'Output')

    # --- Render and Save ---
    # File name includes model type for clarity
    filename = f'{model_name.replace(" ", "_")}_Architecture'
    dot.render(filename, view=False, format='png')

    print(f"✅ Generated detailed visualization saved as '{filename}.png'")
    print(f"   Note: You still need the system 'dot' executable installed (e.g., via 'brew install graphviz').")
    return filename + '.png'


# --- Example Usage for the models you listed ---

# Parameters based on configuration files (as derived in the previous steps):
llama_scout_params = {
    "model_name": "Llama-4-Scout-Instruct",
    "num_layers": 48,  # Estimated based on 17B-16E models
    "is_moe": True,
    "num_experts": 16,  # From the "16E" in the model name
    "has_vision": True
}

gemma_3_params = {
    "model_name": "Gemma-3-27b",
    "num_layers": 62,  # Estimated for 27b
    "is_moe": False,  # Gemma is typically dense
    "num_experts": 0,
    "has_vision": True
}

# Run the visualization function for the models
llama_file = visualize_multimodal_moe(**llama_scout_params)
gemma_file = visualize_multimodal_moe(**gemma_3_params)

# Add a simpler model for comparison
llama_3_params = {
    "model_name": "Llama-3-8B (Dense/Text-Only)",
    "num_layers": 32,
    "is_moe": False,
    "num_experts": 0,
    "has_vision": False
}
llama_3_file = visualize_multimodal_moe(**llama_3_params)

print("\nTo view the files, open the PNG images generated in your script directory.")