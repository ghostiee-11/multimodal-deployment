# data_loader.py
import pandas as pd

def load_and_clean_data(products_csv_path, reviews_csv_path, alldocs_csv_path):
    """
    Loads product, review, and all_documents CSVs into pandas DataFrames,
    and performs basic cleaning like stripping product_id.
    """
    
    df_products = pd.read_csv(products_csv_path)
    df_reviews = pd.read_csv(reviews_csv_path)
    df_alldocs = pd.read_csv(alldocs_csv_path)
    print("CSVs loaded successfully!")
    

    # Basic Cleaning (Handle potential leading/trailing spaces in product_id)
    df_products['product_id'] = df_products['product_id'].str.strip()
    df_reviews['product_id'] = df_reviews['product_id'].str.strip()
    df_alldocs['product_id'] = df_alldocs['product_id'].str.strip()
    
    
    print("\n--- Initial Data Overview ---")
    print(f"Products DataFrame (df_products): Shape {df_products.shape}, Unique Product IDs: {df_products['product_id'].nunique()}")
    print(f"Customer Reviews DataFrame (df_reviews): Shape {df_reviews.shape}, Unique Product IDs: {df_reviews['product_id'].nunique()}")
    print(f"All Documents DataFrame (df_alldocs): Shape {df_alldocs.shape}, Unique Product IDs: {df_alldocs['product_id'].nunique()}")

    # Product ID Consistency Check
    product_ids_products = set(df_products['product_id'].unique())
    product_ids_reviews = set(df_reviews['product_id'].unique())
    product_ids_alldocs = set(df_alldocs['product_id'].unique())

    if product_ids_reviews - product_ids_products:
        print(f"WARNING: {len(product_ids_reviews - product_ids_products)} product_ids in reviews NOT in products master list.")
    if product_ids_alldocs - product_ids_products:
         print(f"WARNING: {len(product_ids_alldocs - product_ids_products)} product_ids in alldocs NOT in products master list.")

    return df_products, df_reviews, df_alldocs

if __name__ == '__main__':


    PRODUCTS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/products_final.csv'
    REVIEWS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/customer_reviews.csv'
    ALL_DOCS_CSV_PATH = '/Users/amankumar/Desktop/Aims/final data/all_documents.csv'

    df_products, df_reviews, df_alldocs = load_and_clean_data(
        PRODUCTS_CSV_PATH,
        REVIEWS_CSV_PATH,
        ALL_DOCS_CSV_PATH
    )

    if df_products is not None:
        print("\nData loading and initial cleaning complete (test run).")
        print("df_products head:\n", df_products.head())