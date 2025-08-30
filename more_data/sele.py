import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

OUTPUT_FILE = "products.csv"

def scrape_category(category_name, url, max_pages=1):
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # run in background
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    results = []

    for page in range(1, max_pages + 1):
        page_url = url if page == 1 else f"{url}?p={page}"
        print(f"Scraping {category_name} - Page {page}")
        driver.get(page_url)
        time.sleep(3)  # wait for JS to load

        products = driver.find_elements(By.CSS_SELECTOR, "li.item.product.product-item")

        for p in products:
            try:
                title = p.find_element(By.CSS_SELECTOR, "a.product-item-link").text.strip()
            except:
                title = None

            try:
                price = p.find_element(By.CSS_SELECTOR, "span.price").text.strip()
            except:
                price = None

            try:
                image_url = p.find_element(By.CSS_SELECTOR, "img.product-image-photo").get_attribute("src")
            except:
                image_url = None

            # drinks.ng doesn’t show ratings
            rating = None

            results.append({
                "Product_Name": title,
                "Product_Price": price,
                "Product_Image_URL": image_url,
                "Product_Category": category_name,
                "Rating": rating
            })

    driver.quit()
    return results


def main():
    categories = {
        "Beverages": "https://drinks.ng/mixers-soft-drinks-fubar",
        # "Wines": "https://drinks.ng/wines-zntvh",
    }

    all_results = []
    for cat, url in categories.items():
        all_results.extend(scrape_category(cat, url, max_pages=2))

    df = pd.DataFrame(all_results, columns=[
        "Product_Name", "Product_Price", "Product_Image_URL",
        "Product_Category", "Rating"
    ])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Done. Saved {len(df)} products to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
