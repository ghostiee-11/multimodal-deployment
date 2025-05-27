# prepare_text_chunks.py
import pandas as pd
import uuid
import os 
from data_loader import load_and_clean_data

# Path to the new OCR CSV file
IMAGE_OCR_CSV_PATH = 'image_ocr_texts_cleaned.csv' 

def create_retrievable_text_df(df_products, df_reviews, df_alldocs):
    """
    Creates a DataFrame of individual text chunks for embedding,
    based on the loaded product, review, all_documents data, and image OCR data.
    """
    if df_products is None or df_reviews is None or df_alldocs is None:
        print("One or more input DataFrames are None. Cannot proceed.")
        return None

    retrievable_texts_list = []

    # 1. Process Customer Reviews (df_reviews)
    print("\nProcessing customer reviews for text chunking...")
    for index, row in df_reviews.iterrows():
        product_id = row['product_id']
        original_doc_id = row['doc_id']
        aspect_name = row['aspect_name']
        overall_sentiment = row['overall_sentiment']

        if pd.notna(row['summary_text']) and str(row['summary_text']).strip():
            retrievable_texts_list.append({
                'text_chunk_id': f"chunk_{uuid.uuid4()}",
                'product_id': product_id,
                'original_doc_id': original_doc_id,
                'source_file': 'customer_reviews.csv',
                'text_type': 'aspect_summary',
                'aspect': aspect_name,
                'sentiment': overall_sentiment,
                'text_content': str(row['summary_text']).strip()
            })

        for i in range(1, 4):
            snippet_col = f'snippet_{i}_text'
            highlight_col = f'snippet_{i}_highlight'
            if pd.notna(row[snippet_col]) and str(row[snippet_col]).strip():
                retrievable_texts_list.append({
                    'text_chunk_id': f"chunk_{uuid.uuid4()}",
                    'product_id': product_id,
                    'original_doc_id': original_doc_id,
                    'source_file': 'customer_reviews.csv',
                    'text_type': 'review_snippet',
                    'aspect': aspect_name,
                    'sentiment': overall_sentiment,
                    'text_content': str(row[snippet_col]).strip(),
                    'highlight': str(row[highlight_col]) if pd.notna(row[highlight_col]) else None
                })
    print(f"Processed {len(df_reviews)} review rows for text chunking.")

    # 2. Process All Documents (df_alldocs)
    print("\nProcessing all_documents for text chunking...")
    for index, row in df_alldocs.iterrows():
        product_id = row['product_id']
        original_doc_id = row['doc_id']
        doc_type = row['doc_type']
        text_content = row['text_content']

        if pd.notna(text_content) and str(text_content).strip():
            current_aspect = 'General'
            if doc_type == 'specification':
                current_aspect = 'Specification'
            elif 'description' in doc_type:
                current_aspect = 'Description'

            retrievable_texts_list.append({
                'text_chunk_id': f"chunk_{uuid.uuid4()}",
                'product_id': product_id,
                'original_doc_id': original_doc_id,
                'source_file': 'all_documents.csv',
                'text_type': doc_type,
                'aspect': current_aspect,
                'sentiment': 'N/A', # Assuming N/A for specs and descriptions
                'text_content': str(text_content).strip()
            })
    print(f"Processed {len(df_alldocs)} all_docs rows for text chunking.")

    # 3. Process Image OCR Texts (image_ocr_texts_cleaned.csv)
    print(f"\nProcessing image OCR texts from {IMAGE_OCR_CSV_PATH}...")
    try:
        df_ocr_texts = pd.read_csv(IMAGE_OCR_CSV_PATH)
        df_ocr_texts.dropna(subset=['product_id', 'image_path', 'ocr_text_cleaned'], inplace=True) # Ensure essential columns are present
        
        for index, row in df_ocr_texts.iterrows():
            ocr_text_cleaned = str(row['ocr_text_cleaned']).strip()
            if ocr_text_cleaned: # Only add if there's cleaned OCR text
                product_id = str(row['product_id']).strip()
                image_path = str(row['image_path']).strip()
                
                retrievable_texts_list.append({
                    'text_chunk_id': f"chunk_{uuid.uuid4()}",
                    'product_id': product_id,
                    'original_doc_id': image_path, # Use image_path as a unique identifier for this OCR text
                    'source_file': IMAGE_OCR_CSV_PATH,
                    'text_type': 'image_ocr_text',
                    'aspect': 'image_content_textual', # Specific aspect for OCR text
                    'sentiment': 'N/A', # OCR text is generally neutral fact
                    'text_content': ocr_text_cleaned,
                    'image_filename': os.path.basename(image_path) # Store filename for easier reference
                })
        print(f"Processed {len(df_ocr_texts)} OCR text rows, adding {sum(1 for item in retrievable_texts_list if item['source_file'] == IMAGE_OCR_CSV_PATH)} valid OCR text chunks.")
    except FileNotFoundError:
        print(f"Warning: OCR text file not found at {IMAGE_OCR_CSV_PATH}. Skipping OCR text integration.")
    except Exception as e:
        print(f"Error processing {IMAGE_OCR_CSV_PATH}: {e}")


    df_retrievable_texts = pd.DataFrame(retrievable_texts_list)
    return df_retrievable_texts

if __name__ == '__main__':
    # Define paths to your CSV files
    PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
    REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv'
    ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'
    # IMAGE_OCR_CSV_PATH is defined globally in the script

    # --- Load data using the imported function ---
    df_products, df_reviews, df_alldocs = load_and_clean_data(
        PRODUCTS_CSV_PATH,
        REVIEWS_CSV_PATH,
        ALL_DOCS_CSV_PATH
    )

    if df_products is not None: # Check if data loading was successful
        # --- Create the retrievable texts DataFrame ---
        df_retrievable_texts = create_retrievable_text_df(df_products, df_reviews, df_alldocs)

        if df_retrievable_texts is not None and not df_retrievable_texts.empty:
            # --- Inspect the new df_retrievable_texts ---
            print("\n--- Retrievable Texts DataFrame (df_retrievable_texts) ---")
            print("Shape:", df_retrievable_texts.shape)
            print("\nHead (showing potential OCR texts):")
            # Show some OCR texts if they exist
            ocr_sample = df_retrievable_texts[df_retrievable_texts['text_type'] == 'image_ocr_text']
            if not ocr_sample.empty:
                print(ocr_sample.head(5))
            else:
                print(df_retrievable_texts.head(5))
                
            print("\nInfo:")
            df_retrievable_texts.info(verbose=True) # verbose=True to see all columns even if many
            print("\nText Types Counts:")
            print(df_retrievable_texts['text_type'].value_counts())
            print("\nAspect Counts (Top 10):")
            print(df_retrievable_texts['aspect'].value_counts().nlargest(10))
            print("\nMissing values per column:\n", df_retrievable_texts.isnull().sum())
            print(f"\nTotal text chunks created: {len(df_retrievable_texts)}")

            # Verify Product ID linkage
            product_ids_in_retrievable = set(df_retrievable_texts['product_id'].unique())
            product_ids_in_master_list = set(df_products['product_id'].unique())
            missing_pids_in_products = product_ids_in_retrievable - product_ids_in_master_list
            if missing_pids_in_products:
                print(f"\nWARNING: {len(missing_pids_in_products)} product_ids found in retrievable texts but NOT in df_products.")
                # print(f"Missing IDs: {list(missing_pids_in_products)[:10]}") # Show a few if many
            else:
                print("\nAll product_ids in retrievable texts are present in df_products. Good!")

            # Save for inspection
            output_filename = 'prepared_text_chunks.csv'
            df_retrievable_texts.to_csv(output_filename, index=False)
            print(f"\nSaved {output_filename} for inspection.")
        else:
            print("Failed to create df_retrievable_texts or it was empty.")
    else:
        print("Data loading failed. Cannot proceed with text chunk preparation.")