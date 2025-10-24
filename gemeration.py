import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load CLIP embedding model
embedding_model = SentenceTransformer("clip-ViT-B-32")

# Load IKEA item metadata (assumed pre-extracted JSON)
with open("ikea_metadata.json", "r") as f:
    ikea_items = json.load(f)

# Example parsed user prompt (simulate LLM output)
parsed_prompt = {
    "room_type": "bedroom",
    "style": "Scandinavian",
    "furniture": ["bed", "wardrobe", "lamp", "rug"],
    "color_palette": ["white", "beige", "light wood"]
}

# Embed prompt features
prompt_embeddings = {
    "style": embedding_model.encode(parsed_prompt["style"], convert_to_tensor=True),
    "colors": embedding_model.encode(" ".join(parsed_prompt["color_palette"]), convert_to_tensor=True)
}

# Match function
def match_furniture(parsed_prompt, ikea_items, prompt_embeddings, top_k=5):
    matches = []
    for item in ikea_items:
        if item["category"] not in parsed_prompt["furniture"]:
            continue

        item_style_embed = embedding_model.encode(item["style"], convert_to_tensor=True)
        item_color_embed = embedding_model.encode(item["color"], convert_to_tensor=True)

        style_score = util.cos_sim(prompt_embeddings["style"], item_style_embed).item()
        color_score = util.cos_sim(prompt_embeddings["colors"], item_color_embed).item()
        combined_score = (0.6 * style_score) + (0.4 * color_score)

        matches.append({
            "name": item["name"],
            "category": item["category"],
            "style": item["style"],
            "color": item["color"],
            "image_path": item.get("image_path", "N/A"),
            "score": combined_score
        })

    matches = sorted(matches, key=lambda x: x["score"], reverse=True)
    return pd.DataFrame(matches[:top_k])

# Run the matcher
results_df = match_furniture(parsed_prompt, ikea_items, prompt_embeddings)
print(results_df)
