import os
import pandas as pd 

root_folder = 'final data'

all_documents = pd.DataFrame()
all_products = pd.DataFrame()


for folder in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder)

    if os.path.isdir(folder_path) and folder.startswith("final_data"):
        doc_path = os.path.join(folder_path, "documents_raw.csv")
        prod_path = os.path.join(folder_path, "products_raw.csv")

        if os.path.exists(doc_path):
            df_doc = pd.read_csv(doc_path)
            df_doc["source_folder"] = folder  # Optional: track origin
            all_documents = pd.concat([all_documents, df_doc], ignore_index=True)

        if os.path.exists(prod_path):
            df_prod = pd.read_csv(prod_path)
            df_prod["source_folder"] = folder  # Optional: track origin
            all_products = pd.concat([all_products, df_prod], ignore_index=True)

all_documents.to_csv("all_documents.csv", index=False)
all_products.to_csv("all_products.csv", index=False)

print("Done: Combined files saved as 'all_documents.csv' and 'all_products.csv'")