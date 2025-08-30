import os
import pandas as pd
import random

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_DIR = os.path.join(BASE_DIR, "egroceryapp", "static", "csv")

MAIN_CSV = os.path.join(CSV_DIR, "nwfilevsc.csv")  # your main product file
OUTPUT_CSV = os.path.join(CSV_DIR, "ECommerce_consumer_behaviour.csv")  # new generated file

# Load main product catalog
products = pd.read_csv(MAIN_CSV)

# Drop Product_detail_id (not needed in customer behaviour file)
products = products.drop(columns=["Product_detail_id"], errors="ignore")

# Generate random transactions
transactions = []
num_orders = 2000  # number of fake orders

for _ in range(num_orders):
    # Randomly choose 2–6 products per order
    order_size = random.randint(2, 6)
    chosen_products = products.sample(order_size)

    for _, row in chosen_products.iterrows():
        transactions.append(row.to_dict())

# Save to new CSV
df_transactions = pd.DataFrame(transactions)
df_transactions.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"✅ Generated {num_orders} fake orders")
print(f"📂 Saved to {OUTPUT_CSV}")
