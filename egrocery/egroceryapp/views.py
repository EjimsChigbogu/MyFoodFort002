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
            data_cleaned["Product_detail_id"] == str(item.prodcut_Id_Add)
        ]
        if not product.empty:
            details.extend(product.to_dict("records"))
    return details


def cart_total_price(user):
    total = 0
    for item in add_To_Cart.objects.filter(user=user):
        total += int(item.subtotal_price or 0)
    return total


def get_recommendations(product, rules_df=None):
    rules_df = rules_df or rules
    if rules_df.empty:
        return []
    filtered = rules_df[(rules_df["confidence"] > 0.5) & (rules_df["lift"] > 1)]
    product_rules = filtered[filtered["antecedents"].apply(lambda x: product in x)]
    return list(set(product_rules["consequents"].explode().tolist()))


def get_similar_products(query, knn_model, vectorizer):
    query_vector = vectorizer.transform([query])
    _, indices = knn_model.kneighbors(query_vector)
    return data_cleaned.iloc[indices[0]]


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

def search_suggestion_for_all(values):
    values_for_recomd = values
    all_recommendations = set()
    matching_products_random_list = []  # Initialize outside the loop

    for category in values_for_recomd:
        category_recommendations = get_recommendations(category)
        all_recommendations.update(category_recommendations)

    # Extract only frozenset values
    recommended_values = set(all_recommendations)
    lists = [
        ast.literal_eval(s.replace("frozenset(", "[").replace(")", "]"))
        for s in recommended_values
    ]
    flattened_list = [item for sublist in lists for item in sublist]
    final_list = [list(item) for item in flattened_list]

    # Convert each sublist to a string
    result_list = ["+".join(sublist) for sublist in final_list]

    # Convert the numpy array to a Python list
    flattened_list1 = (
        result_list if isinstance(result_list, list) else flattened_list.tolist()
    )

    matching_products_random = data_cleaned[
        data_cleaned["Product_Category"].isin(flattened_list1)
    ]

    matching_products_random = matching_products_random.sample(frac=0.5).reset_index(
        drop=True
    )

    matching_products_random_list = matching_products_random.to_dict(orient="records")

    return matching_products_random_list

def search_results_suggestion(request):
    query = request.GET.get("query", "")

    # Search for matching products based on both name and category
    matching_products = data_cleaned[
        (data_cleaned["Product_Name"].str.contains(query, case=False))
        | (data_cleaned["Product_Category"].str.contains(query, case=False))
    ]
    if not matching_products.empty:
        # Get the unique product categories for the matching products
        unique_categories = matching_products["Product_Category"].unique()
        # print(unique_categories)
        matching_products_random = search_suggestion_for_all(unique_categories)
        # Collect recommendations for each unique category
        # print(matching_products_random)

        # Store matching_products_random in the session
        request.session["matching_products_random"] = matching_products_random

    else:
        # If no matching products, set recommendations and matching_products_random to empty
        # recommended_values = set()
        matching_products_random = pd.DataFrame()

    # Get similar products using KNN
    similar_products = get_similar_products(query, knn_model, vectorizer)
    print(similar_products)
    context = {
        "query": query,
        "matching_products_random": matching_products_random,
        "similar_products": similar_products,
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
    similar = get_similar_products(p["Product_Name"], knn_model, vectorizer)
    recommendations = get_recommendations(p["Product_Category"])
    context = {
        "product_name": p["Product_Name"],
        "product_category": p["Product_Category"],
        "product_Price": p["Product_Price"],
        "product_Image_URL": p["Product_Image_URL"],
        "Product_dtl_id": p["Product_detail_id"],
        "Product_dtl_rating": p["rating"],
        "matching_products_random": similar.to_dict("records")
        if not similar.empty
        else [],
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
            if not add_To_Cart.objects.filter(
                user=user, prodcut_Id_Add=productID
            ).exists():
                add_To_Cart.objects.create(
                    user=user,
                    prodcut_Id_Add=productID,
                    quantity=form.cleaned_data["quantity"],
                    product_price=form.cleaned_data["product_price"],
                )
                return JsonResponse({"message": "Item Added Successfully."})
            return JsonResponse({"error": "Item already in Cart."}, status=400)
        return JsonResponse({"error": "Invalid form data"}, status=400)
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
def update_cart_item(request):
    cart_item_id = request.POST.get("cart_item_id")
    new_quantity = request.POST.get("new_quantity")
    try:
        item = add_To_Cart.objects.get(prodcut_Id_Add=cart_item_id, user=request.user)
        item.quantity = new_quantity
        item.save()
        return JsonResponse(
            {
                "success": True,
                "new_subtotal": item.subtotal_price,
                "new_total_price": cart_total_price(request.user),
            }
        )
    except add_To_Cart.DoesNotExist:
        return JsonResponse({"success": False, "error": "Item not found"})


@require_POST
@login_required
def delete_cart_item(request):
    cart_item_id = request.POST.get("cart_item_id")
    try:
        item = add_To_Cart.objects.get(prodcut_Id_Add=cart_item_id, user=request.user)
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
        matching_items = cart_data.filter(prodcut_Id_Add=int(cart_id))

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
    new_order.products = ",".join(str(i.prodcut_Id_Add) for i in cart_items)
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
