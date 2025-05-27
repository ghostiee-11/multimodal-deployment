# main_assistant.py
import os
import pandas as pd
from dotenv import load_dotenv # To load .env file for API keys

# Custom module imports
from data_loader import load_and_clean_data
from retriever import (
    initialize_retriever_resources, 
    retrieve_relevant_chunks, # This is retrieve_and_rerank_text_chunks
    retrieve_relevant_images_from_text
)
from llm_handler import configure_gemini, generate_answer_with_gemini

# --- Configuration ---
PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv' # Needed by data_loader
ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'   # Needed by data_loader
IMAGE_BASE_PATH = '/Users/amankumar/Desktop/Aims/final data/images'
# CAPTIONS_CSV_PATH = 'image_captions.csv' # Captions now primarily from image index metadata

# Global DataFrames
df_products_global = None

def initial_setup():
    """Loads API keys and initializes all necessary resources."""
    print("Initializing RAG system resources...")
    load_dotenv() 

    if not os.getenv("PINECONE_API_KEY"):
        print("CRITICAL ERROR: PINECONE_API_KEY not found in environment variables or .env file.")
        return False
    if not os.getenv("GOOGLE_API_KEY"):
        print("CRITICAL ERROR: GOOGLE_API_KEY not found in environment variables or .env file.")
        return False
        
    try:
        configure_gemini() 
        initialize_retriever_resources() 
        
        global df_products_global
        df_prods_temp, _, _ = load_and_clean_data(
            PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH
        )
        if df_prods_temp is None:
            raise FileNotFoundError(f"Failed to load product data from {PRODUCTS_CSV_PATH}")
        df_products_global = df_prods_temp
        print(f"Product metadata (df_products_global) loaded. Shape: {df_products_global.shape}")
        if df_products_global.shape[0] == 0:
             print(f"WARNING: {PRODUCTS_CSV_PATH} seems to be empty or loaded incorrectly.")

        print("RAG system resources initialized successfully.")
        return True
    except Exception as e:
        print(f"Critical error during RAG system initialization: {e}")
        return False

def assemble_llm_context(retrieved_texts, retrieved_images, max_context_items=3):
    """
    Consolidates text and image retrieval results into a list for the LLM.
    Tries to get diverse products and prioritize text matches, then add unique image matches.
    Ensures captions from direct image search are used for associated images.
    """
    global df_products_global

    if df_products_global is None:
        print("Error: Product metadata (df_products_global) not loaded.")
        return []

    final_context_items = []
    
    # Create a lookup for directly retrieved images and their captions/info
    # Key: full_image_path, Value: dict from retrieved_images
    direct_image_matches_info = {
        img_data['image_path']: img_data 
        for img_data in retrieved_images if img_data.get('image_path')
    }

    # 1. Process text-based results
    processed_text_product_ids = set() # Keep track of products added via text
    if retrieved_texts:
        print("\nProcessing text-based retrieval for LLM context...")
        for text_chunk in retrieved_texts:
            product_id = text_chunk['metadata'].get('product_id')
            if not product_id: continue

            product_info_dict = None
            images_for_this_product_context = []

            product_row_df = df_products_global[df_products_global['product_id'] == product_id]
            if not product_row_df.empty:
                product_row = product_row_df.iloc[0]
                product_info_dict = {
                    "title": product_row.get('title'),
                    "price": product_row.get('price'),
                    "product_type": product_row.get('product_type')
                }
                
                if pd.notna(product_row.get('image_paths')):
                    relative_img_paths = str(product_row.get('image_paths')).split(',')
                    for rel_path in relative_img_paths[:2]: # Show up to 2 primary images for this product
                        img_filename = rel_path.strip()
                        if not img_filename: continue
                        full_path = os.path.join(IMAGE_BASE_PATH, img_filename)
                        
                        caption = "Caption not directly retrieved for this associated image."
                        if full_path in direct_image_matches_info:
                            caption = direct_image_matches_info[full_path]['caption'] 
                        
                        images_for_this_product_context.append({
                            "image_path": full_path, 
                            "caption": caption
                        })
            
            item = {
                "type": "text_derived_context",
                "text_content": text_chunk['text_content'],
                "text_score": text_chunk['score'], 
                "text_metadata_details": text_chunk['metadata'],
                "associated_product_id": product_id,
                "associated_product_info": product_info_dict,
                "associated_images": images_for_this_product_context
            }
            final_context_items.append(item)
            processed_text_product_ids.add(product_id)

    # 2. Add unique, directly retrieved image contexts (especially if they bring new product info)
    if retrieved_images:
        print("\nProcessing direct image retrieval results for LLM context...")
        for img_path, img_data in direct_image_matches_info.items():
            product_id = img_data['product_id']
            
            # If this product_id was already added via text results, skip adding it as a new "image_derived_context"
            # We've already tried to enrich the text_derived_context with this image's caption if it matched.
            if product_id in processed_text_product_ids:
                # Ensure the caption is updated in the text_derived_context if this image was primary
                for ctx_item in final_context_items:
                    if ctx_item.get("associated_product_id") == product_id and ctx_item.get("type") == "text_derived_context":
                        for assoc_img in ctx_item.get("associated_images", []):
                            if assoc_img["image_path"] == img_path:
                                assoc_img["caption"] = img_data['caption'] # Update caption
                                break
                continue 

            # If the product is new, add this image_derived_context
            product_info_dict = None
            if product_id and product_id != 'N/A':
                 product_row_df = df_products_global[df_products_global['product_id'] == product_id]
                 if not product_row_df.empty:
                    product_row = product_row_df.iloc[0]
                    product_info_dict = {"title": product_row.get('title'), "price": product_row.get('price')}
            
            item = {
                "type": "image_derived_context",
                "image_path": img_path,
                "image_caption": img_data['caption'],
                "image_score": img_data['score'], 
                "associated_product_id": product_id,
                "associated_product_info": product_info_dict,
                "associated_images": [{"image_path": img_path, "caption": img_data['caption']}] 
            }
            final_context_items.append(item)

    # Sort by score (descending) - This is heuristic as scores are from different systems
    # Prioritize text_derived_context slightly, then by score
    final_context_items.sort(key=lambda x: (
        x['type'] == 'text_derived_context', 
        x.get('text_score', 0.0) if x['type'] == 'text_derived_context' else x.get('image_score', 0.0)
    ), reverse=True)
    
    return final_context_items[:max_context_items]


if __name__ == '__main__':
    if not initial_setup():
        exit()

    while True:
        user_query = input("\n🛍️ Enter your query (or type 'quit' to exit): ")
        if user_query.lower() == 'quit': break
        if not user_query.strip(): continue

        print(f"\n🔎 Processing your query: '{user_query}'")
        
        initial_candidates_for_rerank = 10 
        final_top_k_after_rerank = 3     
        
        retrieved_texts = retrieve_relevant_chunks(
            user_query, 
            initial_top_k=initial_candidates_for_rerank, 
            final_top_k=final_top_k_after_rerank
        )
        
        top_k_image_results = 3 # Retrieve a few more images
        retrieved_images = retrieve_relevant_images_from_text(user_query, top_k=top_k_image_results)
        
        final_llm_context_list = assemble_llm_context(retrieved_texts, retrieved_images, max_context_items=3) 
        
        if final_llm_context_list:
            print(f"\n📋 --- Assembled Context for LLM (Top {len(final_llm_context_list)} items) ---")
            for i, item in enumerate(final_llm_context_list):
                print(f"\nContext Item {i+1} (Type: {item['type']}):")
                if item.get('associated_product_info'): # Check if product_info exists
                    print(f"  Product: {item['associated_product_info'].get('title', 'N/A')} (ID: {item.get('associated_product_id', 'N/A')})")
                elif item.get('associated_product_id'): # Fallback to just product_id if info is missing
                     print(f"  Product ID: {item.get('associated_product_id', 'N/A')}")

                if item['type'] == 'text_derived_context':
                    score_val = item.get('text_score', 0.0) 
                    print(f"  Text (Rerank Score: {score_val:.4f}): \"{item.get('text_content', '')[:150]}...\"")
                    if item.get('text_metadata_details'):
                        print(f"  Source Aspect: {item['text_metadata_details'].get('aspect', 'N/A')}")
                
                elif item['type'] == 'image_derived_context':
                    score_val = item.get('image_score', 0.0)
                    print(f"  Image (CLIP Score: {score_val:.4f}): {item.get('image_path', 'N/A')}")
                    print(f"  Caption: {item.get('image_caption', 'N/A')}")
                
                # Print primary associated images for this context item
                if item.get('associated_images'): 
                    print("  Visuals (Primary for this context item):")
                    for img_info in item['associated_images'][:1]: # Show only first primary image for this item
                        print(f"    - Image: {img_info.get('image_path', 'N/A')}")
                        print(f"      Caption: {img_info.get('caption', 'N/A')}")
                print("-" * 20)

            print("\n🤖 Sending context to LLM for answer generation...")
            llm_answer = generate_answer_with_gemini(user_query, final_llm_context_list)
            print("\n💡 --- Shopping Assistant's Answer ---")
            print(llm_answer)
        else:
            print("\n🤷 No sufficient context assembled from text or image search to send to the LLM.")
            
    print("\nExiting Visual Shopping Assistant. Goodbye!")