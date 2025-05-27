from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import pandas as pd
import json
import re
from dotenv import load_dotenv

# --- Path Configuration (all files in root) ---
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# --- Import helper modules from root ---
# These modules (llm_handler*.py, retriever*.py, data_loader.py) should be in the SAME directory as app.py
# and their internal imports (if any) should not use a 'utils.' prefix.
import llm_handler as llm_v1
import retriever as retriever_v1
import llm_handler_new as llm_v2
import retriever_new as retriever_v2
import llm_handler_new_one as llm_v3
import retriever_new_one as retriever_v3
from data_loader import load_and_clean_data # Assuming this is the one from your files

# Configure Flask: template_folder tells Flask to look for index.html in APP_ROOT
# static_url_path can be set if you want a specific prefix for static files served by custom route
app = Flask(__name__, template_folder=APP_ROOT)

# --- Global Variables for Data & Initialization Status ---
df_products_global_v1, df_products_global_v2, df_products_global_v3 = None, None, None
df_image_captions_global_v2, df_image_captions_global_v3 = None, None # For V2 and V3 caption/text CSVs
initialized_v1, initialized_v2, initialized_v3 = False, False, False

# --- DATA FILE PATHS (assuming they are in a 'data' subdirectory) ---
DATA_DIR = os.path.join(APP_ROOT, 'final data') # Create a 'data' folder in your root for CSVs
PRODUCTS_CSV_PATH = os.path.join(DATA_DIR, '/Users/amankumar/Desktop/Aims/final data/products_final_with_all_image_paths.csv')
REVIEWS_CSV_PATH = os.path.join(DATA_DIR, '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv')
ALL_DOCS_CSV_PATH = os.path.join(DATA_DIR, '/Users/amankumar/Desktop/Aims/final data/all_documents.csv')
V2_IMAGE_CAPTIONS_CSV_PATH = os.path.join(DATA_DIR, 'image_captions_multiple.csv')
V3_IMAGE_COMBINED_TEXTS_CSV_PATH = os.path.join(DATA_DIR, 'image_combined_blip_ocr_filtered_final.csv')

# --- IMAGE FILE PATHS for PIL (if images are in the root or an 'images' subdir) ---
# If your actual .jpg/.png files are in the root:
IMAGE_DIR_FOR_PIL = APP_ROOT
# If your .jpg/.png files are in an 'images' subdirectory in the root:
# IMAGE_DIR_FOR_PIL = os.path.join(APP_ROOT, 'images')
# Create this 'images' folder and put at least B0863FR3S9_box_image.jpg in it for the V3 LLM test.


# --- Simplified Product Data Loader ---
def load_product_data_generic(products_path):
    try:
        df = pd.read_csv(products_path)
        if 'price' in df.columns:
            df['price_numeric'] = pd.to_numeric(
                df['price'].astype(str).str.replace('₹', '').str.replace(',', ''), errors='coerce'
            )
        else: df['price_numeric'] = pd.NA
        return df
    except FileNotFoundError: print(f"Error: Product CSV {products_path} not found."); return pd.DataFrame()
    except Exception as e: print(f"Error loading {products_path}: {e}"); return pd.DataFrame()

# --- Initialization Functions ---
def initial_setup_v1():
    global df_products_global_v1, initialized_v1
    if initialized_v1: return True
    try:
        print("Attempting V1 Initialization...")
        llm_v1.configure_gemini()
        retriever_v1.initialize_retriever_resources()
        # V1's main_assistant.py uses load_and_clean_data which needs PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH
        # Ensure these are correct.
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        df_products_global_v1 = df_prods_temp
        if df_products_global_v1.empty: print("V1 WARNING: Product data is empty!")
        # V1's main_assistant.py uses a global `df_products_global`. We simulate this.
        retriever_v1.df_products_global = df_products_global_v1 # Make it available to retriever_v1 if it uses this global
        llm_v1.df_products_global = df_products_global_v1 # And llm_v1
        print("V1: RAG system resources initialized.")
        initialized_v1 = True
    except Exception as e: print(f"V1 Init Error: {e}"); initialized_v1 = False; import traceback; traceback.print_exc()
    return initialized_v1

def initial_setup_v2():
    global df_products_global_v2, df_image_captions_global_v2, initialized_v2
    if initialized_v2: return True
    try:
        print("Attempting V2 Initialization...")
        llm_v2.configure_gemini()
        retriever_v2.initialize_retriever_resources()
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        df_products_global_v2 = df_prods_temp
        if df_products_global_v2.empty: print("V2 WARNING: Product data empty!")

        df_image_captions_global_v2 = pd.read_csv(V2_IMAGE_CAPTIONS_CSV_PATH)
        # ... (V2 caption processing from your main_assistant_new.py)
        if 'generated_captions_json' in df_image_captions_global_v2.columns:
            df_image_captions_global_v2['all_captions_list'] = df_image_captions_global_v2['generated_captions_json'].apply(lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else [])
        # ... (rest of V2 caption processing) ...
        if 'primary_caption' not in df_image_captions_global_v2.columns:
            df_image_captions_global_v2['primary_caption'] = df_image_captions_global_v2['all_captions_list'].apply(lambda x: x[0] if x and isinstance(x, list) and len(x) > 0 else "Caption unavailable.")

        # Make DFs available to V2 modules if they expect globals
        retriever_v2.df_products_global = df_products_global_v2
        retriever_v2.df_image_captions_global = df_image_captions_global_v2
        # main_assistant_new.py also used IMAGE_BASE_PATH globally in some functions
        retriever_v2.IMAGE_BASE_PATH = IMAGE_DIR_FOR_PIL # Or adjust as needed by retriever_v2

        print("V2: RAG system resources initialized.")
        initialized_v2 = True
    except Exception as e: print(f"V2 Init Error: {e}"); initialized_v2 = False; import traceback; traceback.print_exc()
    return initialized_v2

def initial_setup_v3():
    global df_products_global_v3, df_image_captions_global_v3, initialized_v3
    if initialized_v3: return True
    try:
        print("Attempting V3 Initialization...")
        llm_v3.configure_gemini()
        retriever_v3.initialize_retriever_resources()
        df_prods_temp, _, _ = load_and_clean_data(PRODUCTS_CSV_PATH, REVIEWS_CSV_PATH, ALL_DOCS_CSV_PATH)
        df_products_global_v3 = df_prods_temp
        if df_products_global_v3.empty: print("V3 WARNING: Product data empty!")

        df_image_captions_global_v3 = pd.read_csv(V3_IMAGE_COMBINED_TEXTS_CSV_PATH)
        # ... (V3 caption/text processing from your main_assistant_new_one.py)
        if 'full_image_path' not in df_image_captions_global_v3.columns: raise ValueError("V3: 'full_image_path' missing")
        if 'generated_texts_json' not in df_image_captions_global_v3.columns: raise ValueError("V3: 'generated_texts_json' missing")
        df_image_captions_global_v3['all_captions_list'] = df_image_captions_global_v3['generated_texts_json'].apply(
            lambda x: json.loads(x) if pd.notna(x) and isinstance(x, str) else []
        )
        def get_primary_v3(lst): # Renamed to avoid conflict
            if lst and isinstance(lst, list) and len(lst) > 0 and isinstance(lst[0], dict) and 'text' in lst[0]: return str(lst[0]['text'])
            return "Caption unavailable."
        df_image_captions_global_v3['primary_caption'] = df_image_captions_global_v3['all_captions_list'].apply(get_primary_v3)

        # Make DFs and paths available to V3 modules if they expect globals
        # This is a workaround for modules written with globals. Best is to refactor them.
        main_assistant_new_one_module = __import__('main_assistant_new_one') # Dynamically import to set attrs
        main_assistant_new_one_module.df_products_global = df_products_global_v3
        main_assistant_new_one_module.df_image_captions_global = df_image_captions_global_v3
        main_assistant_new_one_module.IMAGE_BASE_PATH = IMAGE_DIR_FOR_PIL # Ensure this is used for image loading

        print("V3: RAG system resources initialized.")
        initialized_v3 = True
    except Exception as e: print(f"V3 Init Error: {e}"); initialized_v3 = False; import traceback; traceback.print_exc()
    return initialized_v3

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

# Route to serve static files (CSS, JS) from the root directory
@app.route('/<path:filename>')
def serve_static_from_root(filename):
    if filename in ['style.css', 'script.js']:
        return send_from_directory(APP_ROOT, filename)
    # Add this if you have images in the root and reference them in HTML (not typical)
    # elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.ico')):
    #    return send_from_directory(APP_ROOT, filename)
    return "File not found.", 404


@app.route('/api/chat', methods=['POST'])
def chat():
    print(f"--- Received request for /api/chat (Method: {request.method}) ---")
    data = request.json
    user_query = data.get('query')
    history = data.get('history', [])
    iteration = data.get('iteration', 'v3')
    print(f"Query: '{user_query}', Iteration: {iteration}, History length: {len(history)}")

    answer = "Error: Iteration logic not fully implemented in app.py." # Default error

    try:
        if iteration == 'v1':
            if not initialized_v1 and not initial_setup_v1():
                 return jsonify({'error': 'V1 Initialization failed. Check server logs.'}), 500
            print("Processing V1 logic...")
            # --- V1 Logic (Adapted from main_assistant.py) ---
            # You need to copy the main loop logic from main_assistant.py here,
            # and call its helper functions (map_entities_to_product_ids, etc.)
            # ensuring they use df_products_global_v1 and IMAGE_DIR_FOR_PIL (if V1 needs it).
            main_v1_module = __import__('main_assistant') # To access its functions if not copied directly
            main_v1_module.df_products_global = df_products_global_v1
            main_v1_module.IMAGE_BASE_PATH = IMAGE_DIR_FOR_PIL

            parsed_info_v1 = llm_v1.parse_user_query_with_gemini(user_query) # V1 used simpler parser
            
            # target_pids_v1 = main_v1_module.map_entities_to_product_ids(parsed_info_v1, df_products_global_v1)
            # parsed_constraints_v1 = main_v1_module.parse_constraints_from_features(parsed_info_v1.get("key_features_attributes", []))
            # candidate_df_v1 = main_v1_module.filter_products_by_constraints(df_products_global_v1, parsed_constraints_v1) # Adapt this to take target_pids
            # pinecone_filter_v1 = ...
            # texts_v1 = retriever_v1.retrieve_relevant_chunks(user_query, filter_dict=pinecone_filter_v1)
            # images_v1 = retriever_v1.retrieve_relevant_images_from_text(user_query, filter_dict=pinecone_filter_v1)
            # final_context_v1 = main_v1_module.assemble_llm_context(texts_v1, images_v1) # This used global df_products_global

            # Simplified V1 for demo:
            texts_v1_demo = retriever_v1.retrieve_relevant_chunks(user_query, final_top_k=1)
            context_v1_demo = [{"type": "text_derived_context", "text_content": t['text_content']} for t in texts_v1_demo]
            answer = llm_v1.generate_answer_with_gemini(user_query, context_v1_demo, parsed_info_v1)


        elif iteration == 'v2':
            if not initialized_v2 and not initial_setup_v2():
                 return jsonify({'error': 'V2 Initialization failed. Check server logs.'}), 500
            print("Processing V2 logic...")
            # --- V2 Logic (Adapted from main_assistant_new.py) ---
            main_v2_module = __import__('main_assistant_new')
            main_v2_module.df_products_global = df_products_global_v2
            main_v2_module.df_image_captions_global = df_image_captions_global_v2
            main_v2_module.IMAGE_BASE_PATH = IMAGE_DIR_FOR_PIL


            history_str_v2 = llm_v2.format_conversation_history_for_prompt(history)
            parsed_info_v2 = llm_v2.parse_user_query_with_gemini(user_query, history_str_v2)
            # ... (Full V2 pipeline logic: map_entities, parse_constraints, filter_products, retrieve, rerank, assemble_llm_context)
            # Example call to an adapted assemble function:
            # final_context_v2 = main_v2_module.assemble_llm_context(texts_v2, images_vilt_v2) # This used globals

            # Simplified V2 for demo:
            retrieval_query_v2 = parsed_info_v2.get("rewritten_query_for_retrieval", user_query)
            if retrieval_query_v2 == "N/A": retrieval_query_v2 = user_query
            texts_v2_demo = retriever_v2.retrieve_relevant_chunks(retrieval_query_v2, final_top_k=1)
            context_v2_demo = [{"type": "text_derived_context", "text_content": t['text_content']} for t in texts_v2_demo]
            answer = llm_v2.generate_answer_with_gemini(user_query, context_v2_demo, parsed_info_v2, history_str_v2)


        elif iteration == 'v3':
            if not initialized_v3 and not initial_setup_v3():
                 return jsonify({'error': 'V3 Initialization failed. Check server logs.'}), 500
            print("Processing V3 logic...")
            # --- V3 Logic (Adapted from main_assistant_new_one.py) ---
            # This is the most complex. You set the globals on the imported module earlier.
            main_v3_module = __import__('main_assistant_new_one') # Already imported, attributes set in initial_setup_v3

            history_str_v3 = llm_v3.format_conversation_history_for_prompt(history)
            parsed_query_info_v3 = llm_v3.parse_user_query_with_gemini(user_query, history_str_v3)
            
            intent_v3 = parsed_query_info_v3.get("intent_type", "GENERAL_PRODUCT_SEARCH")
            key_features_v3 = parsed_query_info_v3.get("key_features_attributes", [])
            comparison_entities_v3 = parsed_query_info_v3.get("comparison_entities", [])
            product_entities_v3 = parsed_query_info_v3.get("product_entities", [])
            brand_entities_v3 = parsed_query_info_v3.get("brand_entities", [])
            retrieval_query_base_v3 = parsed_query_info_v3.get("rewritten_query_for_retrieval", user_query)
            if not retrieval_query_base_v3 or retrieval_query_base_v3 == "N/A": retrieval_query_base_v3 = user_query

            final_llm_context_list_v3 = []

            if intent_v3 == "PRODUCT_COMPARISON" and len(comparison_entities_v3) >= 2:
                # ... (Full comparison logic from main_assistant_new_one.py using its functions)
                # Example structure:
                for entity_name_original in comparison_entities_v3:
                    entity_name = str(entity_name_original).strip()
                    # temp_parsed_mapping = {"product_entities": [entity_name], "brand_entities": brand_entities_v3}
                    # pids_entity = main_v3_module.map_entities_to_product_ids(temp_parsed_mapping, df_products_global_v3)
                    # ...
                    # texts_entity = retriever_v3.retrieve_relevant_chunks(...)
                    # images_vilt_entity = retriever_v3.rerank_images_with_vilt(...)
                    # context_items = main_v3_module.assemble_llm_context(texts_entity, images_vilt_entity)
                    # final_llm_context_list_v3.extend(...)
                    pass # Placeholder for actual comparison logic
                if not final_llm_context_list_v3: final_llm_context_list_v3 = [{"type":"text", "text_content":"Comparison demo context"}]


            else: # General search for V3
                target_pids_v3 = main_v3_module.map_entities_to_product_ids(parsed_query_info_v3, df_products_global_v3)
                parsed_constraints_v3 = main_v3_module.parse_constraints_from_features(key_features_v3)
                
                candidate_df_v3 = df_products_global_v3
                if target_pids_v3: candidate_df_v3 = df_products_global_v3[df_products_global_v3['product_id'].isin(target_pids_v3)]
                if not candidate_df_v3.empty:
                    candidate_df_v3 = main_v3_module.filter_products_by_constraints(candidate_df_v3, parsed_constraints_v3)
                
                final_retrieval_pids_v3 = candidate_df_v3['product_id'].tolist() if not candidate_df_v3.empty else target_pids_v3
                pinecone_filter_v3 = {"product_id": {"$in": final_retrieval_pids_v3[:20]}} if final_retrieval_pids_v3 else None
                
                # Build effective query for V3
                semantic_parts_v3 = [retrieval_query_base_v3] + product_entities_v3 + brand_entities_v3
                if parsed_constraints_v3.get('other_features_text',''): semantic_parts_v3.append(parsed_constraints_v3['other_features_text'])
                effective_query_v3 = " ".join(list(dict.fromkeys(filter(None, semantic_parts_v3)))).strip()
                if not effective_query_v3: effective_query_v3 = user_query

                texts_retrieved_v3 = retriever_v3.retrieve_relevant_chunks(effective_query_v3, final_top_k=2, filter_dict=pinecone_filter_v3)
                img_query_v3 = effective_query_v3 + " " + " ".join(parsed_query_info_v3.get("visual_aspects_queried",[]))
                images_clip_v3 = retriever_v3.retrieve_relevant_images_from_text_clip(img_query_v3.strip(), top_k=2, filter_dict=pinecone_filter_v3)
                images_vilt_v3 = retriever_v3.rerank_images_with_vilt(img_query_v3.strip(), images_clip_v3, top_k=1) if images_clip_v3 else []
                
                final_llm_context_list_v3 = main_v3_module.assemble_llm_context(texts_retrieved_v3, images_vilt_v3)

            if not final_llm_context_list_v3 and not history:
                 answer = "I'm sorry, I couldn't find enough specific information for that query (V3)."
            else:
                answer = llm_v3.generate_answer_with_gemini(user_query, final_llm_context_list_v3, parsed_query_info_v3, history_str_v3)
        
        else: # Should not happen if iteration choice is controlled
            return jsonify({'error': 'Invalid iteration selected in API'}), 400

        return jsonify({'answer': answer})

    except Exception as e:
        print(f"--- ERROR in /api/chat (Iteration: {iteration}) ---")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    load_dotenv() # Load .env file for API keys
    print(f"Google API Key Loaded: {'YES' if os.getenv('GOOGLE_API_KEY') else 'NO (Check .env)'}")
    print(f"Pinecone API Key Loaded: {'YES' if os.getenv('PINECONE_API_KEY') else 'NO (Check .env)'}")
    # Make sure to run the specific initial_setup for the iteration you want to test by default
    # For example, to test V3 by default if you go to localhost:5001
    # initial_setup_v3() # Or call this based on some default or environment variable
    app.run(host='0.0.0.0', port=5001, debug=True)