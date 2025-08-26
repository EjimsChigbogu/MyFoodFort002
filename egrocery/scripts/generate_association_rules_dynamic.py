import os
import sys
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add project root to Python path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up Django
os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE", "egrocery.settings"
)
import django

django.setup()

# Now safe to import Django models and utils
from django.conf import settings
from egroceryapp.models import Order
from egroceryapp.utils import load_data_and_train

# Paths to CSV files
csv_file_path = os.path.join(settings.BASE_DIR,"egroceryapp", "static", "csv", "nwfilevsc.csv")
rules_csv = os.path.join(settings.BASE_DIR,"egroceryapp", "static", "csv", "association_rules.csv")

# Ensure CSV folders exist
os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
os.makedirs(os.path.dirname(rules_csv), exist_ok=True)

print("Starting script...")
print(f"Source CSV: {csv_file_path}")
print(f"Rules CSV: {rules_csv}")


def generate_association_rules(transaction_data=None):
    # Load the product dataset
    try:
        data_cleaned = pd.read_csv(csv_file_path, encoding="utf-8")
        data_cleaned = data_cleaned.dropna()
        print(f"Data loaded: {len(data_cleaned)} rows")
    except FileNotFoundError:
        print(f"{csv_file_path} not found! Please make sure the file exists.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Build transactions from Order model if not provided
    if transaction_data is None:
        transaction_data = []
        try:
            orders = Order.objects.all()
            for order in orders:
                if order.products:
                    product_ids = [
                        int(pid) for pid in order.products.split(",") if pid.isdigit()
                    ]
                    categories = (
                        data_cleaned[
                            data_cleaned["Product_detail_id"].isin(product_ids)
                        ]["Product_Category"]
                        .unique()
                        .tolist()
                    )
                    if categories:
                        transaction_data.append(categories)
        except Exception as e:
            print(f"Error fetching orders: {e}")
            # fallback to dummy transactions
            categories = data_cleaned["Product_Category"].unique()
            transaction_data = [
                [cat1, cat2]
                for i, cat1 in enumerate(categories)
                for cat2 in categories[i + 1 :]
            ]

    if not transaction_data:
        print("No transactions available. Exiting.")
        return

    # One-hot encode transactions using TransactionEncoder (faster & safer)
    te = TransactionEncoder()
    transaction_matrix = te.fit(transaction_data).transform(transaction_data)
    transaction_df = pd.DataFrame(transaction_matrix, columns=te.columns_)

    # Generate frequent itemsets
    try:
        frequent_itemsets = apriori(transaction_df, min_support=0.01, use_colnames=True)
        if frequent_itemsets.empty:
            print("No frequent itemsets found. Try lowering min_support.")
            return
    except Exception as e:
        print(f"Error generating frequent itemsets: {e}")
        return

    # Generate association rules
    try:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
        if rules.empty:
            print("No association rules generated. Try adjusting thresholds.")
            return
    except Exception as e:
        print(f"Error generating association rules: {e}")
        return

    # Save rules to CSV
    try:
        rules.to_csv(rules_csv, index=False)
        print(f"Association rules saved to {rules_csv}")
    except Exception as e:
        print(f"Error saving rules CSV: {e}")
        return

    # Reload rules into memory
    load_data_and_train()
    print("Rules loaded into memory successfully.")


if __name__ == "__main__":
    generate_association_rules()
