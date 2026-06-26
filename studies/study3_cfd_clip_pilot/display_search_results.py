from pathlib import Path
from IPython.display import display
from PIL import Image, ImageOps
import numpy as np

def resolve_repo_path(path_like, root=ROOT):
    path = Path(path_like)
    if path.is_absolute():
        return path
    return root / path

def display_search_results(results, path_column="image_path", image_height=180, image_width=120):
    for _, row in results.iterrows():
        score = row.get("clip_score", float("nan"))
        print(
            f"Rank {row.get('rank', row.get('candidate_rank', '?'))} | "
            f"target_id={row.get('target_id', row.get('filler_target_id', '?'))} | "
            f"score={score:.4f}"
        )
        image_path = resolve_repo_path(row[path_column])
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img).convert("RGB")
        img_clean = Image.fromarray(np.array(img, dtype=np.uint8), mode="RGB")
        # display(img_clean)
        display(img_clean.resize((image_height, image_width)))
        # display(Image.open(image_path).resize((image_height, image_width)))
