import pandas as pd
import random
import re

INPUT_FILE =  r"C:\Users\HP\Desktop\MyFoodFort002\egrocery\egroceryapp\static\csv\nwfilevsc.csv"
OUTPUT_FILE = "newnwfilesvsc.csv"

# Load data
df = pd.read_csv(INPUT_FILE)

# 1. Drop old Product_detail_id
if "Product_detail_id" in df.columns:
    df.drop(columns=["Product_detail_id"], inplace=True)

# 2. Remove duplicates (Product_Name + Product_Price)
df.drop_duplicates(subset=["Product_Name", "Product_Price"], keep="first", inplace=True)

# 3. Clean Product_Price
def clean_price(price):
    if pd.isna(price) or price == "":
        # Generate random price between 2,000 and 50,000 (adjust as needed)
        return round(random.uniform(2000, 50000), 2)
    try:
        return round(float(price), 2)
    except:
        # If parsing fails, assign random
        return round(random.uniform(2000, 50000), 2)

df["Product_Price"] = df["Product_Price"].apply(clean_price)

# Format with commas and two decimals
df["Product_Price"] = df["Product_Price"].map(lambda x: f"{x:,.2f}")

# 4. Clean Rating
def clean_rating(r):
    if pd.isna(r):
        return None
    match = re.search(r"(\d+(\.\d+)?)", str(r))
    return match.group(1) if match else None

df["rating"] = df["rating"].apply(clean_rating)

# 5. Reassign Product_detail_id starting from 1
df.insert(0, "Product_detail_id", range(1, len(df) + 1))

# Save cleaned CSV
df.to_csv(OUTPUT_FILE, index=False)

print(f"✅ Cleaning completed. Saved to {OUTPUT_FILE}")
