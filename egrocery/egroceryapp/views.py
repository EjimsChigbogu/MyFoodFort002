import os
import pandas as pd
import hashlib
import ast
import random
import json
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_POST
from django.core.serializers.json import DjangoJSONEncoder
from django.db.models import Sum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .models import *
from .forms import LoginForm, AddToCartForm, ProfileRegisterForm, UserRegisterForm
from .data_loader import load_csv
from .data_loader import data_cleaned


# -------------------------
# Load CSV Data
# -------------------------
product_data_vsc = load_csv("nwfilevsc.csv")
product_data_ecommerce_csv = load_csv("ECommerce_consumer_behaviour.csv")
rules = load_csv("association_rules.csv", preserve_ids=False)

data_cleaned = product_data_vsc.copy()
product_names = data_cleaned["Product_Name"].astype(str)
vectorizer = TfidfVectorizer()
product_vectors = vectorizer.fit_transform(product_names)
knn_model = NearestNeighbors(n_neighbors=24, metric="cosine")
knn_model.fit(product_vectors)


# -------------------------
# Utilities
# -------------------------
def get_cart_details(user):
    cart_data = add_To_Cart.objects.filter(user=user)
    details = []
    for item in cart_data:
        product = data_cleaned[
            data_cleaned["Product_detail_id"] == str(item.product_Id_Add)
        ]
        if not product.empty:
            details.extend(product.to_dict("records"))
    return details


def cart_total_price(user):
    total = 0
    for item in add_To_Cart.objects.filter(user=user):
        total += int(item.subtotal_price or 0)
    return total


# Optional light synonym map you can extend
SYNONYMS = {
    "juicers": "juicer",
    "juice extractor": "juicer",
    "skin-care": "skin care",
    "skincare": "skin care",
}

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[\W_]+", " ", s)          # keep letters/numbers as words
    s = re.sub(r"\s+", " ", s).strip()
    s = SYNONYMS.get(s, s)
    # naive singularization: “juicers” -> “juicer”
    if len(s) > 3 and s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s

def ensure_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "name_norm" not in df.columns:
        df["name_norm"] = df["Product_Name"].fillna("").map(normalize_text)
    if "category_norm" not in df.columns:
        df["category_norm"] = df["Product_Category"].fillna("").map(normalize_text)
    return df

def word_boundary_regex(raw_query: str) -> str:
    # \b around the raw query to avoid partial nonsense matches
    q = raw_query.strip()
    if not q:
        return r"$^"
    return rf"\b{re.escape(q)}\b"


def get_recommendations(product, rules_df=None, current_product_id=None, data=None):
    """
    product: current product category/name to find rules for
    rules_df: dataframe containing association rules
    current_product_id: id of the current product (to exclude from results)
    data: product dataframe (so we can map ids <-> names if needed)
    """
    rules_df = rules_df or rules
    if rules_df.empty:
        return []

    # filter strong rules
    filtered = rules_df[(rules_df["confidence"] > 0.5) & (rules_df["lift"] > 1)]

    # only keep rules where the current product is in the antecedents
    product_rules = filtered[filtered["antecedents"].apply(lambda x: product in x)]

    # get all recommended products
    recommended = list(set(product_rules["consequents"].explode().tolist()))

    # exclude current product name
    recommended = [r for r in recommended if r != product]

    # exclude by id if we have the mapping (optional)
    if current_product_id is not None and data is not None:
        # get the current product row
        current_row = data[data["Product_detail_id"] == current_product_id]
        if not current_row.empty:
            current_name = current_row.iloc[0]["Product_Name"]
            recommended = [r for r in recommended if r != current_name]

    return recommended



def get_similar_products(query, knn_model, vectorizer, current_product_id=None):
    """
    Return KNN similar products, EXCLUDING the exact product being viewed.
    """
    query_str = "" if pd.isna(query) else str(query)
    query_vec = vectorizer.transform([query_str])

    k = getattr(knn_model, "n_neighbors", 24)
    try:
        distances, indices = knn_model.kneighbors(query_vec, n_neighbors=k + 5)
    except Exception:
        distances, indices = knn_model.kneighbors(query_vec)

    idxs = list(indices[0])
    sims = data_cleaned.iloc[idxs].copy()

    # Exclude the product currently being viewed by ID (preferred)
    if current_product_id and "Product_detail_id" in sims.columns:
        sims = sims[sims["Product_detail_id"] != current_product_id]
    else:
        # fallback to name-based exclusion
        q_norm = query_str.strip().lower()
        sims = sims[~sims["Product_Name"].astype(str).str.strip().str.lower().eq(q_norm)]

    # Drop duplicates
    if "Product_detail_id" in sims.columns:
        sims = sims.drop_duplicates(subset=["Product_detail_id"], keep="first")
    else:
        sims = sims.drop_duplicates(subset=["Product_Name", "Product_Image_URL"], keep="first")

    return sims.head(k)



# -------------------------
# Views
# -------------------------
def index(request):
    categories = set(product_data_ecommerce_csv["Product_Category"].unique())
    data_sample = data_cleaned.sample(n=20)

    # Handle anonymous users
    if request.user.is_authenticated:
        cart_dtl_data = get_cart_details(request.user)
        total_price = cart_total_price(request.user)
    else:
        cart_dtl_data = []  # or {} depending on what get_cart_details returns
        total_price = 0

    return render(
        request,
        "index.html",
        {
            "unique_categories": categories,
            "csv_data": data_sample,
            "cart_dtl_data": cart_dtl_data,
            "total_price": total_price,
        },
    )


def suggestion(request):
    return render(request, "suggestion.html")

def search_suggestion_for_all(categories):
    """
    Get recommended products for a list of categories.
    Returns full product rows (list of dicts).
    """
    all_rows = []
    for cat in categories:
        rows = get_recommendations(cat, data=data_cleaned)
        if rows:
            all_rows.extend(rows)

    if not all_rows:
        return []

    df = pd.DataFrame(all_rows)

        # ✅ Only deduplicate if Product_detail_id exists
    if "Product_detail_id" in df.columns:
        df = df.drop_duplicates(subset=["Product_detail_id"])
    else:
        df = df.drop_duplicates()

    # Limit or shuffle if you want (example: show up to 12 random recommendations)
    if len(df) > 12:
        df = df.sample(n=12, random_state=42)

    return df.to_dict(orient="records")


# def search_results_suggestion(request):
#     query = request.GET.get("query", "").strip()

#     # 🔎 Debug: check what columns exist in your data_cleaned dataframe
#     print("COLUMNS IN data_cleaned:", list(data_cleaned.columns))

#     if not query:
#         return render(request, "shop.html", {
#             "query": "",
#             "results": [],
#             "recommended_products": [],
#             "similar_products": [],
#         })

#     # Search for matching products by name or category (case-insensitive)
#     matching_products = data_cleaned[
#         (data_cleaned["Product_Name"].str.contains(query, case=False, na=False)) |
#         (data_cleaned["Product_Category"].str.contains(query, case=False, na=False))
#     ]

#     # Default values
#     results = []
#     recommended_products = []
#     similar_products = []

#     if not matching_products.empty:
#         # Main search results (only the products that match the query)
#         results = matching_products.to_dict("records")

#         # Get unique categories from results
#         unique_categories = matching_products["Product_Category"].dropna().unique()

#         # Recommendations from those categories
#         recommended_products = search_suggestion_for_all(unique_categories)

#         # Remove any recommended product already in results
#         if recommended_products:
#             rec_df = pd.DataFrame(recommended_products)
#             # Exclude already matched products from recommendations (only if ID column exists)
#             if "Product_detail_id" in rec_df.columns and "Product_detail_id" in matching_products.columns:
#                 rec_df = rec_df[~rec_df["Product_detail_id"].isin(matching_products["Product_detail_id"])]
#             recommended_products = rec_df.to_dict("records")

#         # Similar products using KNN (optional: based on first match)
#         first_product = matching_products.iloc[0]
#         sim_df = get_similar_products(
#             first_product["Product_Name"], knn_model, vectorizer,
#             current_product_id=first_product["Product_detail_id"]
#         )
#         if hasattr(sim_df, "to_dict"):
#             similar_products = sim_df.to_dict("records")

#     context = {
#         "query": query,
#         "results": results,                   # ✅ use this in template for main grid
#         "recommended_products": recommended_products,  # ✅ use in sidebar or "you may also like"
#         "similar_products": similar_products, # ✅ optional
#     }

#     return render(request, "shop.html", context)

def search_results_suggestion(request):
    query = request.GET.get("query", "").strip()

    # Debug: show what was searched
    print("SEARCH QUERY:", query)

    # Search for matching products by product name OR category
    matching_products = data_cleaned[
        (data_cleaned["Product_Name"].str.contains(query, case=False, na=False))
        | (data_cleaned["Product_Category"].str.contains(query, case=False, na=False))
    ].copy()

    # If matches found
    if not matching_products.empty:
        # ✅ Always show products in that category (main result)
        unique_categories = matching_products["Product_Category"].unique()
        print("MATCHED CATEGORIES:", unique_categories)

        # Get recommendations for those categories
        rec_products = search_suggestion_for_all(unique_categories)

        # Convert rec_products to DataFrame if needed
        rec_df = pd.DataFrame(rec_products)

        # Avoid excluding all results — just keep recommendations separate
        if not rec_df.empty and "Product_detail_id" in rec_df.columns:
            rec_df = rec_df[~rec_df["Product_detail_id"].isin(matching_products["Product_detail_id"])]

        matching_products_random = rec_df.to_dict(orient="records") if not rec_df.empty else []
    else:
        # No results found
        matching_products = pd.DataFrame()
        matching_products_random = []

    # Get similar products using KNN (optional boost)
    similar_products = get_similar_products(query, knn_model, vectorizer)
    print("SIMILAR PRODUCTS:", similar_products.shape)


    context = {
        "query": query,
        "matching_products": matching_products.to_dict(orient="records"),
        "matching_products_random": matching_products_random,
        "similar_products": similar_products.to_dict(orient="records") if not similar_products.empty else [],
    }

    return render(request, "shop.html", context)


def shop(request):
    return render(
        request,
        "shop.html",
        {
            "data_to_display": data_cleaned.sample(n=48),
            "total_price": cart_total_price(request.user),
        },
    )


def single_product(request, Product_detail_id):
    product = data_cleaned[data_cleaned["Product_detail_id"] == Product_detail_id]
    if product.empty:
        return render(request, "product_not_found.html")

    p = product.iloc[0]

    # 🔥 Pass current product id so it’s excluded from results
    similar = get_similar_products(
        p["Product_Name"], 
        knn_model, 
        vectorizer, 
        current_product_id=p["Product_detail_id"]
    )

    # recommendations = get_recommendations(p["Product_Category"])
    recommendations = get_recommendations(
        p["Product_Category"],
        current_product_id=p["Product_detail_id"],
        data=data_cleaned
    )
    context = {
        "product_name": p["Product_Name"],
        "product_category": p["Product_Category"],
        "product_Price": p["Product_Price"],
        "product_Image_URL": p["Product_Image_URL"],
        "Product_dtl_id": p["Product_detail_id"],
        "Product_dtl_rating": p["rating"],
        "matching_products_random": similar.to_dict("records") if not similar.empty else [],
        "recommendations": recommendations
    }
    return render(request, "single_product.html", context)


def search_results(request):
    query = request.GET.get("query", "")
    recs = get_recommendations(query)
    matches = data_cleaned[data_cleaned["Product_Category"].isin(recs)].sample(frac=1)
    similar = get_similar_products(query, knn_model, vectorizer)
    return render(
        request,
        "search_results.html",
        {
            "query": query,
            "recommendations": recs,
            "matching_products_random": matches,
            "similar_products": similar,
        },
    )


def search_suggestions(request):
    q = request.GET.get("q", "")
    suggestions = data_cleaned[
        data_cleaned["Product_Name"].str.contains(q, case=False)
        | data_cleaned["Product_Category"].str.contains(q, case=False)
    ][["Product_Name", "Product_Category"]][:10]
    return JsonResponse(suggestions.to_dict("records"), safe=False)


# -------------------------
# Authentication
# -------------------------
def loginform(request):
    return render(request, "myaccount.html")


@login_required
def sign_out(request):
    logout(request)
    return redirect("login")


def myaccount(request):
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            user = authenticate(
                request,
                username=form.cleaned_data["username"],
                password=form.cleaned_data["password"],
            )
            if user:
                auth_login(request, user)
                return redirect("index")
    else:
        form = LoginForm()

    # Handle anonymous users safely
    if request.user.is_authenticated:
        total_price = cart_total_price(request.user)
    else:
        total_price = 0  # or whatever default makes sense

    return render(
        request,
        "myaccount.html",
        {"form": form, "total_price": total_price},
    )


def signup(request):
    if request.method == "POST":
        form = UserRegisterForm(request.POST)
        p_form = ProfileRegisterForm(request.POST)
        if form.is_valid() and p_form.is_valid():
            user = form.save()
            Profile.objects.create(user=user, **p_form.cleaned_data)
            auth_login(request, user)
            return redirect("login")
    else:
        form = UserRegisterForm()
        p_form = ProfileRegisterForm()
    return render(request, "signup.html", {"form": form, "p_reg_form": p_form})


# -------------------------
# Cart
# -------------------------
@login_required
@ensure_csrf_cookie
def add_to_cart_dbs(request, productID):
    if request.method == "POST":
        form = AddToCartForm(request.POST)
        if form.is_valid():
            user = request.user
            quantity = form.cleaned_data.get("quantity")
            price = form.cleaned_data.get("product_price")

            try:
                cart_item = add_To_Cart.objects.get(user=user, product_Id_Add=productID)  # FIX typo
                cart_item.quantity += quantity
                cart_item.product_price = price
                cart_item.save()
                return JsonResponse({"message": "Item quantity updated successfully."})
            except add_To_Cart.DoesNotExist:
                add_To_Cart.objects.create(
                    user=user,
                    product_Id_Add=productID,
                    quantity=quantity,
                    product_price=price,
                )
                return JsonResponse({"message": "Item added successfully."})
        else:
            return JsonResponse({"error": form.errors}, status=400)
    return JsonResponse({"error": "Invalid request"}, status=400)



@login_required
def initiate_payment(request):
    if request.method == "POST":
        discount_data = Discount_data.objects.get(user=request.user)
        grand_totalPrice = discount_data.grand_total
        amount = grand_totalPrice * 100
        email = request.user.email
        name = f"{request.user.first_name} {request.user.last_name}"

        print(email, name)

        client = razorpay.Client(
            auth=(settings.RAZORPAY_API_KEY, settings.RAZORPAY_API_SECRET)
        )

        payment_data = {
            "amount": amount,
            "currency": "INR",
            "receipt": "order_receipt",
            "notes": {
                "email": email,
            },
        }

        order = client.order.create(data=payment_data)
        print(order)
        # Include key, name, description, and image in the JSON response
        response_data = {
            "id": order["id"],
            "amount": order["amount"],
            "currency": order["currency"],
            "key": settings.RAZORPAY_API_KEY,
            "name": "Pandit E-grocery",
            "description": "Payment for Your Product",
            "image": "https://yourwebsite.com/logo.png",  # Replace with your logo URL
        }

        return JsonResponse(response_data)

    return render(request, "cart.html")


def payment_success(request):
    return render(request, "payment_success.html")


def payment_failed(request):
    return render(request, "payment_failed.html")


@login_required
def cart(request):
    cart_data = get_cart_details(request.user)
    return render(
        request,
        "cart.html",
        {
            "cart_dtl_data": cart_data,
            "total_price": cart_total_price(request.user),
        },
    )

@require_POST
@login_required
def update_cart_item(request, item_id):
    if request.method == "POST":
        try:
            cart_item = add_To_Cart.objects.get(id=item_id, user=request.user)
            data = json.loads(request.body)
            new_quantity = int(data.get("quantity", 1))

            if new_quantity > 0:
                cart_item.quantity = new_quantity
                # 👇 Recalculate subtotal based on product price × quantity
                cart_item.subtotal_price = cart_item.product_price * cart_item.quantity
                cart_item.save()
                return JsonResponse({
                    "message": "Cart updated successfully",
                    "new_subtotal": cart_item.subtotal_price,
                })
            else:
                cart_item.delete()
                return JsonResponse({"message": "Item removed from cart"})

        except add_To_Cart.DoesNotExist:
            return JsonResponse({"error": "Cart item not found"}, status=404)

    return JsonResponse({"error": "Invalid request"}, status=400)


@require_POST
@login_required
def delete_cart_item(request):
    cart_item_id = request.POST.get("cart_item_id")
    try:
        item = add_To_Cart.objects.get(product_Id_Add=cart_item_id, user=request.user)
        item.delete()
        return JsonResponse({"success": True})
    except add_To_Cart.DoesNotExist:
        return JsonResponse({"success": False, "error": "Item not found"})


@login_required
def get_cart_data(request):
    # Assuming you have a function to get the user's cart data
    cart_dtl_data = get_cart_details(request.user)
    total_price = cart_total_price(request.user)
    cart_data = add_To_Cart.objects.filter(user=request.user)

    context_list = []

    for value in cart_dtl_data:
        cart_id = value["Product_detail_id"]
        matching_items = cart_data.filter(product_Id_Add=int(cart_id))

        for item in matching_items:
            context_list.append(
                {
                    "item": {
                        "id": item.id,
                        "subtotal_price": item.subtotal_price,
                        "quantity": item.quantity,
                        # Add other fields as needed
                    },
                    "detail": {
                        "Product_Name": value[
                            "Product_Name"
                        ],  # Assuming 'Product_Name' is the key
                        # Add other fields from 'value' as needed
                    },
                }
            )

    # print(context_list)

    # Do not convert the list to a JSON string here
    # DjangoJSONEncoder will take care of serialization in JsonResponse
    context_json = context_list

    # Process context_json as needed

    return JsonResponse(
        {"cart_data": context_json, "total_price": total_price},
        encoder=DjangoJSONEncoder,
    )


# -------------------------
# Static Pages
# -------------------------
def about(request):
    return render(request, "about.html")


def contact(request):
    return render(request, "contact.html")


# -------------------------
# Orders
# -------------------------
@login_required
def handle_successful_purchase(request):
    user = request.user
    new_order = Order.objects.create(user=user, payment_status="Paid")
    cart_items = add_To_Cart.objects.filter(user=user)
    new_order.total_items_placed = (
        cart_items.aggregate(Sum("quantity"))["quantity__sum"] or 0
    )
    new_order.total_price = (
        cart_items.aggregate(Sum("subtotal_price"))["subtotal_price__sum"] or 0
    )
    new_order.products = ",".join(str(i.product_Id_Add) for i in cart_items)
    new_order.product_All_quantity = ",".join(str(i.quantity) for i in cart_items)
    new_order.save()
    cart_items.delete()
    return render(request, "purchase_success.html")


@login_required
def order_history(request):
    orders = Order.objects.filter(user=request.user)
    order_details = []
    for order in orders:
        if order.products:
            p_ids = [int(pid) for pid in order.products.split(",") if pid.isdigit()]
            filtered = data_cleaned[data_cleaned["Product_detail_id"].isin(p_ids)]
            quantities = (
                [int(q) for q in order.product_All_quantity.split(",") if q.isdigit()]
                if order.product_All_quantity
                else None
            )
            products = (
                [
                    {
                        "product_detail": filtered[
                            filtered["Product_detail_id"] == pid
                        ].iloc[0],
                        "quantity": qty,
                    }
                    for pid, qty in zip(p_ids, quantities)
                ]
                if quantities
                else []
            )
            order_details.append({"order": order, "products": products})
    return render(request, "order_history.html", {"order_details": order_details})
