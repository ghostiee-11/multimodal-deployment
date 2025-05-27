import os
import pandas as pd
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import json # To store list of captions as a JSON string in CSV

# --- Configuration ---
VALID_IMAGES_CSV_PATH = 'valid_product_images.csv'
CAPTION_MODEL_ID = "Salesforce/blip-image-captioning-large"
OUTPUT_CAPTIONS_CSV = 'image_captions_multiple.csv' # New output file

# Global model and processor
caption_processor = None
caption_model = None
caption_device = None

def load_valid_image_data_for_captioning():
    # (Same as your existing function)
    try:
        df = pd.read_csv(VALID_IMAGES_CSV_PATH)
        df_unique_images = df[['full_image_path', 'product_id']].drop_duplicates(subset=['full_image_path']).reset_index(drop=True)
        print(f"Loaded {VALID_IMAGES_CSV_PATH}, found {len(df_unique_images)} unique images for captioning.")
        return df_unique_images
    except FileNotFoundError:
        print(f"Error: Valid image data CSV not found at {VALID_IMAGES_CSV_PATH}")
        return None

def initialize_captioning_model():
    # (Same as your existing function)
    global caption_processor, caption_model, caption_device
    if caption_model is None or caption_processor is None:
        print(f"Loading image captioning model: {CAPTION_MODEL_ID}...")
        caption_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {caption_device}")
        try:
            caption_processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
            caption_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID).to(caption_device)
            caption_model.eval()
            print("Image captioning model and processor loaded.")
        except Exception as e:
            print(f"Error loading captioning model: {e}")
            raise

def is_caption_valid(caption_text, min_length=8, max_word_repetitions=4, min_unique_words_ratio=0.3):
    if not caption_text or len(caption_text) < min_length:
        return False
    words = caption_text.lower().split()
    if not words:
        return False
    
    # Check for excessive repetition of the same word
    from collections import Counter
    word_counts = Counter(words)
    if any(count > max_word_repetitions for count in word_counts.values() if len(words) > max_word_repetitions + 2):
        if word_counts.most_common(1)[0][1] / len(words) > 0.6 and len(words) > 5:
            # print(f"Filtering caption due to high repetition of a single word: '{caption_text}'")
            return False
            
    # Check for captions that are just one word repeated many times
    if len(set(words)) / len(words) < min_unique_words_ratio and len(words) > 5 : # e.g. "marshall marshall marshall..."
        # print(f"Filtering caption due to low unique word ratio: '{caption_text}'")
        return False
    
    # Avoid "an image of" or "a picture of" if it's too dominant and caption too short
    if caption_text.lower().startswith(("an image of", "a picture of", "a close up of")) and len(words) < 7:
        # print(f"Filtering caption due to generic start and short length: '{caption_text}'")
        return False
        
    return True

def generate_multiple_captions_for_image(image_path):
    global caption_processor, caption_model, caption_device

    if caption_model is None or caption_processor is None:
        raise RuntimeError("Captioning model not initialized.")

    try:
        pil_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Warning: Could not open image {image_path}. Error: {e}")
        return []

    generated_captions = set() # Use a set to store unique captions

    # --- Strategy 1: Different decoding parameters ---
    # Strategy 1.1: Default (often beam search)
    inputs_default = caption_processor(pil_image, return_tensors="pt").to(caption_device)
    with torch.no_grad():
        outputs_default = caption_model.generate(**inputs_default, max_length=74, num_beams=5, early_stopping=True)
    cap_default = caption_processor.decode(outputs_default[0], skip_special_tokens=True).strip()
    if is_caption_valid(cap_default): generated_captions.add(cap_default)

    # Strategy 1.2: Nucleus sampling (top_p)
    inputs_nucleus = caption_processor(pil_image, return_tensors="pt").to(caption_device)
    with torch.no_grad():
        outputs_nucleus = caption_model.generate(
            **inputs_nucleus, max_length=74, do_sample=True, top_p=0.9, num_return_sequences=1 #num_beams=1 for sampling
        )
    cap_nucleus = caption_processor.decode(outputs_nucleus[0], skip_special_tokens=True).strip()
    if is_caption_valid(cap_nucleus): generated_captions.add(cap_nucleus)
    
    # Strategy 1.3: Top-k sampling
    inputs_topk = caption_processor(pil_image, return_tensors="pt").to(caption_device)
    with torch.no_grad():
        outputs_topk = caption_model.generate(
            **inputs_topk, max_length=74, do_sample=True, top_k=50, num_return_sequences=1 #num_beams=1 for sampling
        )
    cap_topk = caption_processor.decode(outputs_topk[0], skip_special_tokens=True).strip()
    if is_caption_valid(cap_topk): generated_captions.add(cap_topk)


    # --- Strategy 2: Prompt Engineering (Conditional Captioning) ---
    prompts = [
        "a detailed description of the product:",
        "features of this product:",
        # "this product is used for:" # Might be too inferential for BLIP
    ]
    for prompt in prompts:
        if len(generated_captions) >= 5: break # Limit total captions
        try:
            inputs_prompted = caption_processor(pil_image, text=prompt, return_tensors="pt").to(caption_device)
            with torch.no_grad():
                outputs_prompted = caption_model.generate(**inputs_prompted, max_length=74, num_beams=3)
            cap_prompted = caption_processor.decode(outputs_prompted[0], skip_special_tokens=True).strip()
            if is_caption_valid(cap_prompted) and cap_prompted.lower() != prompt.lower() : # Ensure it's not just repeating the prompt
                 generated_captions.add(cap_prompted)
        except Exception as e:
            print(f"  Warning: Error during prompted captioning for '{prompt}': {e}")
            continue
            
    return list(generated_captions)[:5] # Return up to 5 unique, valid captions

if __name__ == '__main__':
    df_images_to_caption = load_valid_image_data_for_captioning()

    if df_images_to_caption is not None and not df_images_to_caption.empty:
        try:
            initialize_captioning_model()
        except Exception as e:
            print(f"Failed to initialize captioning model: {e}")
            exit()

        print(f"\nStarting multiple caption generation for {len(df_images_to_caption)} images...")
        start_total_time = time.time()
        
        results = []
        for index, row in df_images_to_caption.iterrows():
            img_path = row['full_image_path']
            product_id = row['product_id']
            
            print(f"  Processing image {index + 1}/{len(df_images_to_caption)}: {os.path.basename(img_path)}")
            start_img_time = time.time()
            captions_list = generate_multiple_captions_for_image(img_path) # Now returns a list
            end_img_time = time.time()
            
            if captions_list:
                print(f"    Generated {len(captions_list)} captions (took {end_img_time - start_img_time:.2f}s):")
                # for i_cap, cap_text in enumerate(captions_list): print(f"      {i_cap+1}. {cap_text}")

                results.append({
                    'full_image_path': img_path,
                    'product_id': product_id,
                    'generated_captions_json': json.dumps(captions_list) # Store as JSON string
                })
            else:
                print(f"    Failed to generate any valid captions for {os.path.basename(img_path)}")
                results.append({
                    'full_image_path': img_path,
                    'product_id': product_id,
                    'generated_captions_json': json.dumps([])
                })
        
        end_total_time = time.time()
        print(f"\nFinished caption generation for all images in {end_total_time - start_total_time:.2f} seconds.")

        df_captions = pd.DataFrame(results)
        df_captions.to_csv(OUTPUT_CAPTIONS_CSV, index=False)
        print(f"\nSaved multiple image captions to {OUTPUT_CAPTIONS_CSV}")
        print(df_captions.head())
        
        print("\nStep for Multiple Image Caption Generation Complete.")
    else:
        print("No valid image data loaded for captioning.")