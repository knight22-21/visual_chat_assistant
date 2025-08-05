# backend/blip_infer.py

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")  # Put your HF API token in .env as HF_TOKEN=...

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "Salesforce/blip2-flan-t5-xl"

processor = Blip2Processor.from_pretrained(model_name, use_auth_token=HF_TOKEN)
model = Blip2ForConditionalGeneration.from_pretrained(model_name, use_auth_token=HF_TOKEN).to(device)

def get_caption(image_path: str) -> str:
    try:
        image = Image.open(image_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt").to(device)

        # Generate captions with max length and beam search
        generated_ids = model.generate(**inputs, max_new_tokens=50, num_beams=5)
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)

        return caption

    except Exception as e:
        return f"Error generating caption: {e}"

def caption_multiple_frames(frame_dir: str, frame_sample_rate: int = 1):
    frame_paths = sorted(Path(frame_dir).glob("*.jpg"))
    selected_frames = frame_paths[::frame_sample_rate]

    results = []
    for frame_path in selected_frames:
        caption = get_caption(str(frame_path))
        results.append({
            "frame": frame_path.name,
            "caption": caption
        })
    return results
