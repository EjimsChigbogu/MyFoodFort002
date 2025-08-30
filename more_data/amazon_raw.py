import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# --- Config ---
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

OUTPUT_FILE = "products.csv"


def scrape_category(category_name, url, max_pages=1):
    results = []

    for page in range(1, max_pages + 1):
        print(f"Scraping {category_name} - Page {page}")
        page_url = url if page == 1 else f"{url}?page={page}"

        response = requests.get(page_url, headers=HEADERS)
        if response.status_code != 200:
            print(f"❌ Failed to fetch {page_url} (status {response.status_code})")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        # TODO: Adjust selectors depending on site structure
        products = soup.find_all("div", class_="product-item-info")

        for product in products:
            # Product name
            title_tag = product.find("a", class_="product-item-link")
            title = title_tag.get_text(strip=True) if title_tag else None

            # Price
            price_tag = product.find("span", class_="price")
            price = price_tag.get_text(strip=True) if price_tag else None

            # Image
            image_tag = product.find("img")
            image_url = image_tag["src"] if image_tag and image_tag.has_attr("src") else None

            # Rating (optional, many sites don't have)
            rating_tag = product.find("span", class_="rating")
            rating = rating_tag.get_text(strip=True) if rating_tag else None

            results.append({
                "Product_Name": title,
                "Product_Price": price,
                "Product_Image_URL": image_url,
                "Product_Category": category_name,
                "Rating": rating
            })

        time.sleep(random.uniform(2, 4))  # be polite

    return results


def main():
    # Add your categories + links here
    categories = {
        "Beverages": "https://drinks.ng/mixers-soft-drinks-fubar",
        # "Wines": "https://drinks.ng/wines-zntvh",
        # "Fresh Fruits": "https://example.com/fruits",
    }

    all_results = []
    for cat, url in categories.items():
        all_results.extend(scrape_category(cat, url, max_pages=1))

    df = pd.DataFrame(all_results, columns=[
        "Product_Name", "Product_Price", "Product_Image_URL",
        "Product_Category", "Rating"
    ])

    # Save to CSV
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done. Saved {len(df)} products to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
