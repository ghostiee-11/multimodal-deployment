# main_assistant_w.py (Refactored for Iteration 1)

import os
import pandas as pd
from dotenv import load_dotenv
import re

# V1 Custom module imports
from data_loader import load_and_clean_data
from retriever import (
    initialize_retriever_resources as initialize_retriever_resources_v1,
    retrieve_relevant_chunks as retrieve_relevant_chunks_v1,
    retrieve_relevant_images_from_text as retrieve_relevant_images_from_text_v1
)
from llm_handler import (
    configure_gemini as configure_gemini_v1,
    generate_answer_with_gemini as generate_answer_with_gemini_v1,
    parse_user_query_with_gemini as parse_user_query_with_gemini_v1
)

# --- V1 Configuration ---
PRODUCTS_CSV_PATH_V1 = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH_V1 = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv'
ALL_DOCS_CSV_PATH_V1 = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'
IMAGE_BASE_PATH_V1 = '/Users/amankumar/Desktop/Aims/final data/images/'

df_products_global_v1 = None
MAX_CONVERSATION_HISTORY_TURNS_V1 = 3 # V1 LLM handler might not use history for answer generation

def initial_setup_v1(): # Renamed for clarity
    global df_products_global_v1
    print("Server (Iteration 1): Initializing V1 RAG system resources...")
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"): print("V1 CRITICAL ERROR: PINECONE_API_KEY not found."); return False
    if not os.getenv("GOOGLE_API_KEY"): print("V1 CRITICAL ERROR: GOOGLE_API_KEY not found."); return False
    
    try:
        configure_gemini_v1()
        initialize_retriever_resources_v1()
        
        df_prods_temp, _, _ = load_and_clean_data(
            PRODUCTS_CSV_PATH_V1, REVIEWS_CSV_PATH_V1, ALL_DOCS_CSV_PATH_V1
        )
        if df_prods_temp is None: raise FileNotFoundError(f"V1 Error: Failed to load product data from {PRODUCTS_CSV_PATH_V1}")
        df_products_global_v1 = df_prods_temp
        print(f"V1: Product metadata (df_products_global_v1) loaded. Shape: {df_products_global_v1.shape}")
        
        if 'price' in df_products_global_v1.columns:
            df_products_global_v1['price_numeric'] = pd.to_numeric(
                df_products_global_v1['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
        else: df_products_global_v1['price_numeric'] = pd.NA
        print("V1: RAG system resources initialized successfully.")
        return True
    except Exception as e:
        print(f"V1 Critical Error: During RAG system initialization: {e}"); return False

def map_entities_to_product_ids_v1(parsed_query_info, df_all_products):
    if df_all_products is None or df_all_products.empty: return []
    product_entities = [str(e).lower().strip() for e in parsed_query_info.get("product_entities", []) if str(e).strip()]
    brand_entities = [str(b).lower().strip() for b in parsed_query_info.get("brand_entities", []) if str(b).strip()]
    candidate_pids = set()
    if 'title' not in df_all_products.columns: return []
    
    df_all_products_temp = df_all_products.copy()
    df_all_products_temp['title_lower_temp'] = df_all_products_temp['title'].astype(str).str.lower()

    if product_entities:
        for entity_model_name in product_entities:
            search_terms = [term for term in re.split(r'\s+|-', entity_model_name) if len(term) > 1] 
            if not search_terms: continue
            
            condition = pd.Series([True] * len(df_all_products_temp))
            for term in search_terms: 
                condition &= df_all_products_temp['title_lower_temp'].str.contains(re.escape(term), na=False, regex=True)
            
            entity_has_brand = False
            if brand_entities:
                for brand in brand_entities:
                    if brand in entity_model_name:
                        condition &= df_all_products_temp['title_lower_temp'].str.contains(re.escape(brand), na=False, regex=True)
                        entity_has_brand = True
                        break
            if not entity_has_brand and brand_entities:
                brand_match_condition = pd.Series([False] * len(df_all_products_temp))
                for brand in brand_entities: 
                    brand_match_condition |= df_all_products_temp['title_lower_temp'].str.contains(re.escape(brand), na=False, regex=True)
                if brand_match_condition.any():
                    condition &= brand_match_condition
            
            matches = df_all_products_temp[condition]
            for pid in matches['product_id'].tolist(): candidate_pids.add(pid)
    elif brand_entities and not product_entities: 
        for brand in brand_entities:
            matches = df_all_products_temp[df_all_products_temp['title_lower_temp'].str.contains(re.escape(brand), na=False, regex=True)]
            for pid in matches['product_id'].tolist(): candidate_pids.add(pid)
    
    return list(candidate_pids)


def parse_constraints_from_features_v1(key_features_attributes):
    constraints = {"price_min": None, "price_max": None, "color": None, "type": None, "other_features_text": ""}
    other_features_list = []
    color_keywords = ["blue","black","red","white","grey","gray","green","silver","beige","aqua","crimson","purple","pink","yellow","orange"]
    type_keywords = ["over-ear","on-ear","in-ear","neckband","earbuds","tws"]
    for feature in key_features_attributes:
        feature_lower = str(feature).lower()
        price_max_match = re.search(r"(?:price under|less than|below|max price|upto|maximum|under rs)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        price_min_match = re.search(r"(?:price over|more than|above|min price|minimum|starting at|starting from)\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        if price_max_match: constraints["price_max"] = float(price_max_match.group(1).replace(',', ''))
        elif price_min_match: constraints["price_min"] = float(price_min_match.group(1).replace(',', ''))
        elif feature_lower.startswith("color:"): constraints["color"] = feature_lower.split(":", 1)[1].strip()
        elif feature_lower.startswith("type:"): constraints["type"] = feature_lower.split(":", 1)[1].strip()
        else:
            found_color = next((c for c in color_keywords if re.search(r'\b' + re.escape(c) + r'\b', feature_lower)), None)
            found_type = next((t for t in type_keywords if re.search(r'\b' + re.escape(t) + r'\b', feature_lower)), None)
            if found_color and not constraints["color"]: constraints["color"] = found_color
            elif found_type and not constraints["type"]: constraints["type"] = found_type
            else: other_features_list.append(feature)
    constraints["other_features_text"] = " ".join(other_features_list).strip()
    return constraints

def filter_products_by_constraints_v1(df_prods, constraints_dict):
    if df_prods is None or df_prods.empty: return pd.DataFrame(columns=df_prods.columns if df_prods is not None else [])
    filtered_df = df_prods.copy()
    if 'price_numeric' not in filtered_df.columns: print("V1 Warning: 'price_numeric' column not found for filtering.")
    else:
        if constraints_dict.get("price_min") is not None: filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] >= constraints_dict["price_min"])]
        if constraints_dict.get("price_max") is not None: filtered_df = filtered_df[filtered_df['price_numeric'].notna() & (filtered_df['price_numeric'] <= constraints_dict["price_max"])]
    if constraints_dict.get("color") and 'title' in filtered_df.columns: 
        color_regex = r'\b' + re.escape(constraints_dict["color"].lower()) + r'\b'
        filtered_df = filtered_df[filtered_df['title'].astype(str).str.lower().str.contains(color_regex, na=False, regex=True)]
    if constraints_dict.get("type") and 'title' in filtered_df.columns: 
        type_regex = r'\b' + re.escape(constraints_dict["type"].lower()) + r'\b'
        type_condition = pd.Series([False] * len(filtered_df), index=filtered_df.index)
        if 'product_type' in filtered_df.columns and filtered_df['product_type'].notna().any(): 
            type_condition |= filtered_df['product_type'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        type_condition |= filtered_df['title'].astype(str).str.lower().str.contains(type_regex, na=False, regex=True)
        filtered_df = filtered_df[type_condition]
    return filtered_df


def get_product_details_and_images_v1(product_id, direct_image_matches_lookup):
    global df_products_global_v1, IMAGE_BASE_PATH_V1
    product_info_dict = None; images_info_list = []
    if not product_id or df_products_global_v1 is None: return product_info_dict, images_info_list
    product_row_df = df_products_global_v1[df_products_global_v1['product_id'] == product_id]
    if product_row_df.empty: return product_info_dict, images_info_list
    product_row = product_row_df.iloc[0]
    product_info_dict = {"title": product_row.get('title'), "price": product_row.get('price'), "product_type": product_row.get('product_type')}
    added_image_paths = set()

    for img_path_lookup, img_data_lookup in direct_image_matches_lookup.items():
        if len(images_info_list) >= 1: break
        if img_data_lookup.get('product_id') == product_id:
            images_info_list.append({
                "image_path": img_data_lookup['image_path'], 
                "caption": img_data_lookup.get('caption', "Image related to product.")
            })
            added_image_paths.add(img_data_lookup['image_path'])

    if not images_info_list and pd.notna(product_row.get('image_paths')):
        general_image_filenames_str = str(product_row.get('image_paths'))
        first_img_filename = general_image_filenames_str.split(',')[0].strip()
        if first_img_filename:
            full_path = os.path.join(IMAGE_BASE_PATH_V1, first_img_filename)
            if full_path not in added_image_paths and os.path.exists(full_path):
                caption_for_llm = f"Primary image for {product_info_dict.get('title', 'the product')}."
                if full_path in direct_image_matches_lookup:
                    caption_for_llm = direct_image_matches_lookup[full_path].get('caption', caption_for_llm)
                images_info_list.append({"image_path": full_path, "caption": caption_for_llm})
                
    return product_info_dict, images_info_list


def assemble_llm_context_v1(retrieved_texts, retrieved_images, max_total_context_items=3):
    global df_products_global_v1
    if df_products_global_v1 is None: print("V1 Error: Product metadata not loaded."); return []
    final_context_items = []
    direct_image_matches_lookup = {img['image_path']: img for img in retrieved_images if img.get('image_path')}
    processed_pids = set()

    if retrieved_texts:
        for text_chunk in retrieved_texts[:max_total_context_items]: 
            if len(final_context_items) >= max_total_context_items: break
            pid = text_chunk['metadata'].get('product_id')
            if not pid: continue
            info, images = get_product_details_and_images_v1(pid, direct_image_matches_lookup)
            final_context_items.append({
                "type": "text_derived_context", "text_content": text_chunk['text_content'],
                "text_score": text_chunk.get('score', 0.0), "text_metadata_details": text_chunk['metadata'],
                "associated_product_id": pid, "associated_product_info": info,
                "associated_images": images
            })
            processed_pids.add(pid)

    if retrieved_images:
        images_to_add_count = max_total_context_items - len(final_context_items)
        added_img_items = 0
        for img_data in sorted(retrieved_images, key=lambda x: x.get('score', 0.0), reverse=True):
            if added_img_items >= images_to_add_count: break
            pid = img_data['product_id']
            if pid in processed_pids:
                enriched = False
                for ctx_item in final_context_items:
                    if ctx_item.get("associated_product_id") == pid and ctx_item.get("type") == "text_derived_context":
                        if not any(assoc_img.get("image_path") == img_data["image_path"] for assoc_img in ctx_item.get("associated_images", [])):
                             if len(ctx_item.get("associated_images",[])) < 1:
                                ctx_item.setdefault("associated_images", []).append({"image_path": img_data["image_path"], "caption": img_data["caption"]})
                                enriched = True; break
                if enriched: continue
            
            info, _ = get_product_details_and_images_v1(pid, {})
            final_context_items.append({
                "type": "image_derived_context", "image_path": img_data['image_path'],
                "image_caption": img_data['caption'], "image_score": img_data['score'],
                "associated_product_id": pid, "associated_product_info": info,
                "associated_images": [{"image_path": img_data['image_path'], "caption": img_data['caption']}]
            })
            if pid: processed_pids.add(pid)
            added_img_items +=1
            
    def sort_key_v1(item):
        priority = 0; score = 0.0
        if item['type'] == 'text_derived_context': priority = 0; score = item.get('text_score', 0.0)
        elif item['type'] == 'image_derived_context': priority = 1; score = item.get('image_score', 0.0)
        return (priority, -score)
    final_context_items.sort(key=sort_key_v1)
    return final_context_items[:max_total_context_items]


def get_assistant_response(user_query_original, conversation_history_tuples): # Renamed
    global df_products_global_v1
    if df_products_global_v1 is None:
        return "Iteration 1 Error: Product data not loaded."

    print(f"\n🔎 V1 Assistant: Processing query: '{user_query_original}'")
    parsed_query_info = parse_user_query_with_gemini_v1(user_query_original) # V1 parser
    
    intent = parsed_query_info.get("intent_type", "general_search")
    key_features_attributes = parsed_query_info.get("key_features_attributes", [])
    comparison_entities = parsed_query_info.get("comparison_entities", [])
    
    final_llm_context_list = []

    if intent == "product_comparison" and len(comparison_entities) == 2:
        max_items_per_comp_product = 2 
        for entity_raw_name in comparison_entities:
            entity_name = str(entity_raw_name).strip()
            pids_for_this_entity = map_entities_to_product_ids_v1({"product_entities": [entity_name]}, df_products_global_v1)
            entity_filter = {"product_id": {"$in": pids_for_this_entity[:10]}} if pids_for_this_entity else None
            retrieval_query_for_entity = f"{entity_name} {' '.join(key_features_attributes)}".strip()
            if not retrieval_query_for_entity.replace(entity_name, "").strip():
                retrieval_query_for_entity = f"{entity_name} details specifications"
            
            texts_entity = retrieve_relevant_chunks_v1(retrieval_query_for_entity, initial_top_k=5, final_top_k=max_items_per_comp_product, filter_dict=entity_filter)
            images_entity = retrieve_relevant_images_from_text_v1(retrieval_query_for_entity, top_k=1, filter_dict=entity_filter) 
            context_entity_items = assemble_llm_context_v1(texts_entity, images_entity, max_total_context_items=max_items_per_comp_product)
            
            if context_entity_items:
                final_llm_context_list.append({"type": "comparison_intro", "product_name": entity_name})
                final_llm_context_list.extend(context_entity_items)
    else:
        target_pids_from_entities = map_entities_to_product_ids_v1(parsed_query_info, df_products_global_v1)
        parsed_constraints = parse_constraints_from_features_v1(key_features_attributes)
        
        candidate_products_df = df_products_global_v1
        if any(v is not None for k,v in parsed_constraints.items() if k != 'other_features_text') : 
            candidate_products_df = filter_products_by_constraints_v1(df_products_global_v1, parsed_constraints)
        
        final_retrieval_pids = []
        if candidate_products_df is not None and not candidate_products_df.empty: # Check if df is not None
            if target_pids_from_entities: 
                final_retrieval_pids = list(set(target_pids_from_entities) & set(candidate_products_df['product_id'].tolist()))
                if not final_retrieval_pids: 
                    final_retrieval_pids = candidate_products_df['product_id'].tolist() if not candidate_products_df.empty else target_pids_from_entities
            else: 
                final_retrieval_pids = candidate_products_df['product_id'].tolist()
        elif target_pids_from_entities:
             final_retrieval_pids = target_pids_from_entities
        
        pinecone_filter = {"product_id": {"$in": final_retrieval_pids[:20]}} if final_retrieval_pids else None
        
        semantic_query_parts = [parsed_constraints.get('other_features_text','')] + parsed_query_info.get("product_entities", []) + parsed_query_info.get("brand_entities", [])
        # Ensure original query is part of semantic search if not fully covered by parsed parts
        current_semantic_text = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip().lower()
        if user_query_original.lower() not in current_semantic_text:
             semantic_query_parts.append(user_query_original)

        retrieval_query_effective = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip()
        if not retrieval_query_effective: retrieval_query_effective = user_query_original

        texts = retrieve_relevant_chunks_v1(retrieval_query_effective, initial_top_k=10, final_top_k=3, filter_dict=pinecone_filter)
        images = retrieve_relevant_images_from_text_v1(retrieval_query_effective, top_k=2, filter_dict=pinecone_filter) 
        final_llm_context_list = assemble_llm_context_v1(texts, images, max_total_context_items=3)

    if final_llm_context_list:
        # V1's generate_answer_with_gemini might not take conversation_history_str in its llm_handler.py
        llm_answer = generate_answer_with_gemini_v1(user_query_original, final_llm_context_list, parsed_query_info)
        return llm_answer
    else:
        return "I'm sorry, I couldn't find enough specific information for Iteration 1 to answer that."