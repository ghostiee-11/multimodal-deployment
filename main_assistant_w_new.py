# main_assistant_w_new.py (Refactored for Iteration 2)

import os
import pandas as pd
from dotenv import load_dotenv
import re
from rapidfuzz import process, fuzz, utils as rapidfuzz_utils
import json

# V2 Custom module imports
from data_loader import load_and_clean_data
from retriever_new import (
    initialize_retriever_resources as initialize_retriever_resources_v2,
    retrieve_relevant_chunks as retrieve_relevant_chunks_v2,
    retrieve_relevant_images_from_text_clip as retrieve_relevant_images_from_text_clip_v2,
    rerank_images_with_vilt as rerank_images_with_vilt_v2
)
from llm_handler_new import (
    configure_gemini as configure_gemini_v2,
    generate_answer_with_gemini as generate_answer_with_gemini_v2,
    parse_user_query_with_gemini as parse_user_query_with_gemini_v2,
    format_conversation_history_for_prompt as format_conversation_history_for_prompt_v2
)

# --- V2 Configuration ---
PRODUCTS_CSV_PATH_V2 = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
REVIEWS_CSV_PATH_V2 = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv'
ALL_DOCS_CSV_PATH_V2 = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'
IMAGE_BASE_PATH_V2 = '/Users/amankumar/Desktop/Aims/final data/images/'
IMAGE_CAPTIONS_CSV_PATH_V2 = 'image_captions_multiple.csv'

df_products_global_v2 = None
df_image_captions_global_v2 = None
MAX_CONVERSATION_HISTORY_TURNS_V2 = 3

def initial_setup_v2(): # Ensure this name matches server.py import
    global df_products_global_v2, df_image_captions_global_v2
    print("Server (Iteration 2): Initializing V2 RAG system resources...")
    load_dotenv()
    if not os.getenv("PINECONE_API_KEY"): print("V2 CRITICAL ERROR: PINECONE_API_KEY not found."); return False
    if not os.getenv("GOOGLE_API_KEY"): print("V2 CRITICAL ERROR: GOOGLE_API_KEY not found."); return False
    
    try:
        configure_gemini_v2()
        initialize_retriever_resources_v2()
        
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH_V2, REVIEWS_CSV_PATH_V2, ALL_DOCS_CSV_PATH_V2)
        if df_prods_temp is None: raise FileNotFoundError(f"V2 Error: Failed to load product data from {PRODUCTS_CSV_PATH_V2}")
        df_products_global_v2 = df_prods_temp
        print(f"V2: Product metadata (df_products_global_v2) loaded. Shape: {df_products_global_v2.shape}")
        
        if 'price' in df_products_global_v2.columns:
            df_products_global_v2['price_numeric'] = pd.to_numeric(
                df_products_global_v2['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
        else: df_products_global_v2['price_numeric'] = pd.NA

        try:
            df_image_captions_global_v2 = pd.read_csv(IMAGE_CAPTIONS_CSV_PATH_V2)
            if 'full_image_path' not in df_image_captions_global_v2.columns or \
               'generated_captions_json' not in df_image_captions_global_v2.columns:
                raise ValueError(f"'full_image_path' or 'generated_captions_json' missing in {IMAGE_CAPTIONS_CSV_PATH_V2}")
            
            df_image_captions_global_v2['all_captions_list'] = df_image_captions_global_v2['generated_captions_json'].apply(
                lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
            )
            df_image_captions_global_v2['primary_caption'] = df_image_captions_global_v2['all_captions_list'].apply(
                lambda x: x[0] if x and isinstance(x, list) and len(x) > 0 else "Caption unavailable."
            )
            print(f"V2: Image captions (df_image_captions_global_v2 from {IMAGE_CAPTIONS_CSV_PATH_V2}) loaded. Shape: {df_image_captions_global_v2.shape}")
        except FileNotFoundError:
            print(f"V2 Warning: Image captions CSV '{IMAGE_CAPTIONS_CSV_PATH_V2}' not found.")
            df_image_captions_global_v2 = pd.DataFrame(columns=['full_image_path', 'product_id', 'primary_caption', 'all_captions_list'])
        except Exception as e_cap:
            print(f"V2 Error loading/processing V2 image captions: {e_cap}")
            df_image_captions_global_v2 = pd.DataFrame(columns=['full_image_path', 'product_id', 'primary_caption', 'all_captions_list'])

        print("V2: RAG system resources initialized successfully.")
        return True
    except Exception as e:
        print(f"V2 Critical Error: During RAG system initialization: {e}"); return False

def map_entities_to_product_ids_v2(parsed_query_info, df_all_products):
    if df_all_products is None or df_all_products.empty: return []
    product_entities_raw = parsed_query_info.get("product_entities", [])
    brand_entities_raw = parsed_query_info.get("brand_entities", [])
    product_entities = [str(e).lower().strip() for e in product_entities_raw if str(e).strip()]
    brand_entities = [str(b).lower().strip() for b in brand_entities_raw if str(b).strip()]
    candidate_pids = set()
    if 'title' not in df_all_products.columns: return []
    
    temp_df = df_all_products[['product_id', 'title']].copy()
    temp_df['title_lower'] = temp_df['title'].astype(str).str.lower()
    choices_titles_only = temp_df['title_lower'].tolist()
    product_id_list_for_index = temp_df['product_id'].tolist()

    if not choices_titles_only : return []
    if product_entities:
        for entity_model_name in product_entities:
            processed_entity = rapidfuzz_utils.default_process(entity_model_name)
            best_match = process.extractOne(processed_entity, choices_titles_only, scorer=fuzz.WRatio, score_cutoff=80) 
            if best_match:
                match_title_lower, score, index = best_match
                pid_candidate = product_id_list_for_index[index]
                entity_brands_lower = [b for b in brand_entities if b in entity_model_name]
                if not entity_brands_lower: entity_brands_lower = brand_entities
                brand_confirmed = True
                if entity_brands_lower: brand_confirmed = any(brand in match_title_lower for brand in entity_brands_lower)
                if brand_confirmed: candidate_pids.add(pid_candidate)
    if not candidate_pids and brand_entities: 
        for brand_name in brand_entities:
            for idx, row in temp_df.iterrows():
                if brand_name in row['title_lower']: candidate_pids.add(row['product_id'])
    return list(candidate_pids)

def parse_constraints_from_features_v2(key_features_attributes):
    constraints = {"price_min": None, "price_max": None, "color": None, "type": None, "other_features_text": ""}
    other_features_list = []
    color_keywords = ["blue","black","red","white","grey","gray","green","silver","beige","aqua","crimson","purple","pink","yellow","orange", "gold", "brown", "cream"]
    type_keywords = ["over-ear","on-ear","in-ear","neckband","earbuds","tws", "true wireless", "wired", "wireless"]
    
    for feature in key_features_attributes:
        feature_lower = str(feature).lower().strip()
        if not feature_lower: continue
        price_max_match = re.search(r"(?:price\s*(?:under|less than|below|max|upto|maximum)|under\s*(?:rs|₹))\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        price_min_match = re.search(r"(?:price\s*(?:over|more than|above|min|minimum)|(?:starting|from)\s*(?:rs|₹))\s*₹?\s*(\d[\d,]*\.?\d*)", feature_lower)
        if price_max_match: constraints["price_max"] = float(price_max_match.group(1).replace(',', ''))
        elif price_min_match: constraints["price_min"] = float(price_min_match.group(1).replace(',', ''))
        elif feature_lower.startswith("color:"): constraints["color"] = feature_lower.split(":", 1)[1].strip()
        elif feature_lower.startswith("type:"): constraints["type"] = feature_lower.split(":", 1)[1].strip()
        else:
            found_color = next((c for c in color_keywords if re.search(r'\b' + re.escape(c) + r'\b', feature_lower)), None)
            found_type = next((t for t in type_keywords if re.search(r'\b' + re.escape(t) + r'\b', feature_lower)), None)
            if found_color and not constraints["color"]: constraints["color"] = found_color
            elif found_type and not constraints["type"]: constraints["type"] = found_type
            else: other_features_list.append(feature.strip())
    constraints["other_features_text"] = " ".join(other_features_list).strip()
    return constraints

def filter_products_by_constraints_v2(df_prods, constraints_dict):
    if df_prods is None or df_prods.empty: return pd.DataFrame(columns=df_prods.columns if df_prods is not None else [])
    filtered_df = df_prods.copy()
    if 'price_numeric' not in filtered_df.columns: print("V2 Warning: 'price_numeric' column not found for filtering.")
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

def get_captions_for_image_path_v2(image_full_path):
    global df_image_captions_global_v2
    default_primary = "Caption unavailable."
    default_all_list_str = []
    if df_image_captions_global_v2 is None or df_image_captions_global_v2.empty or 'full_image_path' not in df_image_captions_global_v2.columns:
        return default_primary, default_all_list_str
    
    caption_row_df = df_image_captions_global_v2[df_image_captions_global_v2['full_image_path'] == image_full_path]
    if not caption_row_df.empty:
        caption_row = caption_row_df.iloc[0]
        all_captions_list_str = caption_row.get('all_captions_list', default_all_list_str)
        primary_caption = caption_row.get('primary_caption', default_primary)
        
        if not (isinstance(all_captions_list_str, list) and all(isinstance(s, str) for s in all_captions_list_str)):
            all_captions_list_str = [str(c) for c in all_captions_list_str] if isinstance(all_captions_list_str, list) else [str(all_captions_list_str)] if pd.notna(all_captions_list_str) else []
        return primary_caption, all_captions_list_str
    return default_primary, default_all_list_str

def get_product_details_and_associated_images_v2(product_id, direct_image_matches_lookup):
    global df_products_global_v2, IMAGE_BASE_PATH_V2
    product_info_dict = None; images_info_list = []
    if not product_id or df_products_global_v2 is None: return product_info_dict, images_info_list
    
    product_row_df = df_products_global_v2[df_products_global_v2['product_id'] == product_id]
    if product_row_df.empty: return product_info_dict, images_info_list
    
    product_row = product_row_df.iloc[0]
    product_info_dict = {"title": product_row.get('title'), "price": product_row.get('price'), "product_type": product_row.get('product_type')}
    added_image_paths = set()

    for img_path_lookup, img_data_lookup in direct_image_matches_lookup.items():
        if len(images_info_list) >= 2: break
        if img_data_lookup.get('product_id') == product_id:
            images_info_list.append({
                "image_path": img_data_lookup['image_path'], 
                "filename": os.path.basename(img_data_lookup['image_path']),
                "primary_caption": img_data_lookup.get('primary_caption', "N/A"),
                "all_captions": img_data_lookup.get('all_captions', []) # list of strings
            })
            added_image_paths.add(img_data_lookup['image_path'])

    if len(images_info_list) < 2 and pd.notna(product_row.get('image_paths')):
        general_image_filenames_str = str(product_row.get('image_paths'))
        for rel_path in general_image_filenames_str.split(','):
            if len(images_info_list) >= 2: break
            img_filename = rel_path.strip()
            if not img_filename: continue
            full_path = os.path.join(IMAGE_BASE_PATH_V2, img_filename)
            if full_path not in added_image_paths and os.path.exists(full_path):
                primary_cap, all_caps_list_str = get_captions_for_image_path_v2(full_path)
                images_info_list.append({
                    "image_path": full_path, "filename": img_filename,
                    "primary_caption": primary_cap, "all_captions": all_caps_list_str
                })
                added_image_paths.add(full_path)
    return product_info_dict, images_info_list


def assemble_llm_context_v2(retrieved_texts, vilt_reranked_images, max_total_context_items=4):
    global df_products_global_v2
    if df_products_global_v2 is None: print("V2 Main Error: Product metadata not loaded."); return []
    final_context_items = []
    
    vilt_images_lookup = {}
    for img_data_vilt in vilt_reranked_images:
        if img_data_vilt.get('image_path'):
            vilt_images_lookup[img_data_vilt['image_path']] = img_data_vilt

    processed_product_ids = set()
    if retrieved_texts:
        for text_chunk in retrieved_texts:
            if len(final_context_items) >= max_total_context_items: break
            pid = text_chunk['metadata'].get('product_id')
            if not pid: continue
            product_details, associated_images_for_text = get_product_details_and_associated_images_v2(pid, vilt_images_lookup)
            final_context_items.append({"type": "text_derived_context", "text_content": text_chunk['text_content'], 
                                      "text_score": text_chunk.get('score', 0.0), "text_metadata_details": text_chunk['metadata'], 
                                      "associated_product_id": pid, "associated_product_info": product_details, 
                                      "associated_images": associated_images_for_text })
            processed_product_ids.add(pid)

    if vilt_reranked_images:
        for img_data in vilt_reranked_images: 
            if len(final_context_items) >= max_total_context_items: break
            pid = img_data.get('product_id'); img_path = img_data.get('image_path')
            if not pid or not img_path: continue
            if pid not in processed_product_ids:
                product_details, _ = get_product_details_and_associated_images_v2(pid, {}) 
                final_context_items.append({"type": "image_derived_context", "image_path": img_path, 
                                          "primary_caption": img_data.get('primary_caption', "N/A"),
                                          "all_captions": img_data.get('all_captions', []),
                                          "image_score": img_data.get('score', 0.0), 
                                          "vilt_score": img_data.get('vilt_score'), 
                                          "associated_product_id": pid, "associated_product_info": product_details,
                                          "associated_images": [{"image_path": img_path, "filename": os.path.basename(img_path),
                                                               "primary_caption": img_data.get('primary_caption', "N/A"),
                                                               "all_captions": img_data.get('all_captions', [])}]})
                processed_product_ids.add(pid)
    return final_context_items[:max_total_context_items]


def get_assistant_response(user_query_original, conversation_history_tuples):
    global df_products_global_v2, MAX_CONVERSATION_HISTORY_TURNS_V2
    
    print(f"\n🔎 V2 Assistant: Processing query: '{user_query_original}' with history_len: {len(conversation_history_tuples)}")
    history_for_prompt_str = format_conversation_history_for_prompt_v2(conversation_history_tuples, MAX_CONVERSATION_HISTORY_TURNS_V2)
    
    query_for_processing = user_query_original
    parsed_query_info = parse_user_query_with_gemini_v2(query_for_processing, history_for_prompt_str)
    
    intent = parsed_query_info.get("intent_type", "GENERAL_PRODUCT_SEARCH")
    key_features_attributes = parsed_query_info.get("key_features_attributes", [])
    comparison_entities = parsed_query_info.get("comparison_entities", [])
    product_entities_from_llm = parsed_query_info.get("product_entities", [])
    brand_entities_from_llm = parsed_query_info.get("brand_entities", [])
    
    retrieval_query_base = parsed_query_info.get("rewritten_query_for_retrieval", query_for_processing)
    if not retrieval_query_base or retrieval_query_base == "N/A":
        retrieval_query_base = query_for_processing

    final_llm_context_list = []
    max_items_per_comp_product = 2 
    max_items_general_query = 3

    if intent == "PRODUCT_COMPARISON" and len(comparison_entities) >= 2:
        for entity_name_original in comparison_entities:
            entity_name = str(entity_name_original).strip()
            temp_parsed_for_entity_mapping = {"product_entities": [entity_name], "brand_entities": brand_entities_from_llm}
            pids_for_this_entity = map_entities_to_product_ids_v2(temp_parsed_for_entity_mapping, df_products_global_v2)
            parsed_constraints_for_query = parse_constraints_from_features_v2(key_features_attributes)
            
            if pids_for_this_entity and df_products_global_v2 is not None:
                temp_df_for_entity_constraints = df_products_global_v2[df_products_global_v2['product_id'].isin(pids_for_this_entity)]
                if not temp_df_for_entity_constraints.empty:
                    filtered_pids_by_constraint = filter_products_by_constraints_v2(temp_df_for_entity_constraints, parsed_constraints_for_query)['product_id'].tolist()
                    if filtered_pids_by_constraint: pids_for_this_entity = filtered_pids_by_constraint
            
            entity_pinecone_filter = {"product_id": {"$in": pids_for_this_entity[:10]}} if pids_for_this_entity else None
            retrieval_query_for_entity_parts = [entity_name] + key_features_attributes + [parsed_constraints_for_query.get('other_features_text','')]
            retrieval_query_for_entity = " ".join(filter(None,retrieval_query_for_entity_parts)).strip().replace("  ", " ")
            if not retrieval_query_for_entity.replace(entity_name,"").strip() : retrieval_query_for_entity = f"{entity_name} details features"

            texts_entity = retrieve_relevant_chunks_v2(retrieval_query_for_entity, initial_top_k=5, final_top_k=max_items_per_comp_product, filter_dict=entity_pinecone_filter)
            images_clip_entity = retrieve_relevant_images_from_text_clip_v2(retrieval_query_for_entity, top_k=3, filter_dict=entity_pinecone_filter) 
            images_vilt_entity = rerank_images_with_vilt_v2(retrieval_query_for_entity, images_clip_entity, top_k=1) if images_clip_entity else []
            context_entity_items = assemble_llm_context_v2(texts_entity, images_vilt_entity, max_total_context_items=max_items_per_comp_product)
            
            if context_entity_items:
                final_llm_context_list.append({"type": "comparison_intro", "product_name": entity_name})
                final_llm_context_list.extend(context_entity_items)
    else: 
        target_pids_from_entities = map_entities_to_product_ids_v2(parsed_query_info, df_products_global_v2)
        parsed_constraints = parse_constraints_from_features_v2(key_features_attributes) 
        candidate_products_df_for_filter = df_products_global_v2
        
        if target_pids_from_entities:
            candidate_products_df_for_filter = df_products_global_v2[df_products_global_v2['product_id'].isin(target_pids_from_entities)]
        
        if candidate_products_df_for_filter is not None and not candidate_products_df_for_filter.empty:
             candidate_products_df_for_filter = filter_products_by_constraints_v2(candidate_products_df_for_filter, parsed_constraints)

        final_retrieval_pids = []
        if candidate_products_df_for_filter is not None and not candidate_products_df_for_filter.empty:
            final_retrieval_pids = candidate_products_df_for_filter['product_id'].tolist()
        elif not target_pids_from_entities and (parsed_constraints["price_min"] or parsed_constraints["price_max"] or parsed_constraints["color"] or parsed_constraints["type"]):
             pass 
        elif target_pids_from_entities: 
             final_retrieval_pids = target_pids_from_entities
        
        pinecone_filter = {"product_id": {"$in": final_retrieval_pids[:20]}} if final_retrieval_pids else None
        
        semantic_query_parts = [retrieval_query_base]
        for pe_part in product_entities_from_llm:
            if pe_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(pe_part)
        for be_part in brand_entities_from_llm:
            if be_part.lower() not in retrieval_query_base.lower(): semantic_query_parts.append(be_part)
        other_f_text = parsed_constraints.get('other_features_text','')
        if other_f_text and other_f_text.lower() not in retrieval_query_base.lower(): 
            semantic_query_parts.append(other_f_text)
            
        retrieval_query_effective = " ".join(list(dict.fromkeys(filter(None, semantic_query_parts)))).strip()
        if not retrieval_query_effective: retrieval_query_effective = query_for_processing

        texts_retrieved = retrieve_relevant_chunks_v2(retrieval_query_effective, initial_top_k=10, final_top_k=max_items_general_query, filter_dict=pinecone_filter)
        image_retrieval_query_final = retrieval_query_effective
        if parsed_query_info.get("visual_aspects_queried"):
            image_retrieval_query_final += " " + " ".join(parsed_query_info["visual_aspects_queried"])
        
        images_clip_retrieved = retrieve_relevant_images_from_text_clip_v2(image_retrieval_query_final.strip(), top_k=5, filter_dict=pinecone_filter) 
        images_vilt_reranked = rerank_images_with_vilt_v2(image_retrieval_query_final.strip(), images_clip_retrieved, top_k=2) if images_clip_retrieved else []
        final_llm_context_list = assemble_llm_context_v2(texts_retrieved, images_vilt_reranked, max_total_context_items=max_items_general_query)
    
    if final_llm_context_list or (intent == "FOLLOW_UP_QUESTION" and conversation_history_tuples):
        llm_answer_english = generate_answer_with_gemini_v2(
            query_for_processing, 
            final_llm_context_list, 
            parsed_query_info,
            conversation_history_str=history_for_prompt_str
        )
        return llm_answer_english
    else:
        return "I'm sorry, I couldn't find enough specific information for Iteration 2 to answer that."

# Removed original if __name__ == '__main__': block