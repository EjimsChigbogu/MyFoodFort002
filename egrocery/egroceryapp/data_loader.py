import os
import pandas as pd
import hashlib

# Base path to your CSV folder
CSV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "egroceryapp", "static", "csv"
)


def load_csv(filename, preserve_ids=True, id_column="Product_detail_id"):
    """
    Safely load a CSV file, normalize columns, and optionally preserve product IDs.
    Returns a cleaned pandas DataFrame.
    """
    csv_path = os.path.join(CSV_DIR, filename)

    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found at {csv_path}")
        return pd.DataFrame()  # return empty DataFrame if missing

    # Load CSV
    df = pd.read_csv(csv_path, encoding="utf-8")

    # Ensure consistent column names
    column_mapping = {
        "title": "Product_Name",
        "name": "Product_Name",
        "product_name": "Product_Name",
        "category": "Product_Category",
        "price": "Product_Price",
        "image_url": "Product_Image_URL",
        "rating": "rating",
    }

    df.rename(columns=column_mapping, inplace=True)

    # If Product_detail_id is missing or needs stable generation
    if preserve_ids and id_column not in df.columns:
        # Generate stable ID based on Product_Name + Product_Image_URL
        df[id_column] = df.apply(
            lambda row: hashlib.md5(
                f"{row.get('Product_Name', '')}{row.get('Product_Image_URL', '')}".encode()
            ).hexdigest(),
            axis=1,
        )

    return df


# Load your main CSV once at module level
data_cleaned = load_csv("nwfilevsc.csv", preserve_ids=True)
