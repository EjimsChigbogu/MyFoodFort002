# this script generates dummy orders and user for association rules to be generated 
import os
import sys
import django
import random
import pandas as pd
from django.contrib.auth import get_user_model

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Django setup
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "egrocery.settings")
django.setup()

from egroceryapp.models import Order

# Path to your products CSV
CSV_PATH = os.path.join(PROJECT_ROOT, "egroceryapp", "static", "csv", "nwfilevsc.csv")


def seed_orders(num_orders=50, min_items=2, max_items=5, num_users=5):
    User = get_user_model()

    # ✅ Create 5 dummy users if they don't already exist
    users = []
    for i in range(1, num_users + 1):
        user, created = User.objects.get_or_create(
            username=f"dummyuser{i}",
            defaults={
                "email": f"dummy{i}@example.com",
                "password": "test1234"
            }
        )
        users.append(user)

    # Load product IDs from CSV
    df = pd.read_csv(CSV_PATH)
    if "Product_detail_id" not in df.columns:
        print("⚠️ 'Product_detail_id' column not found in CSV")
        return

    product_ids = df["Product_detail_id"].dropna().astype(int).tolist()
    if not product_ids:
        print("⚠️ No product IDs found in CSV")
        return

    # Create dummy orders for random users
    for _ in range(num_orders):
        user = random.choice(users)
        items = random.sample(product_ids, random.randint(min_items, max_items))
        order = Order.objects.create(
            user=user,
            products=",".join(map(str, items))
        )
        print(f"✅ Created Order {order.id} for user {user.username} with products {order.products}")


if __name__ == "__main__":
    seed_orders()
