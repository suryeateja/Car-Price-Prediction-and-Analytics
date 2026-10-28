
import requests
from bs4 import BeautifulSoup
import re

url = "https://www.cars24.com/buy-used-honda-cars-hyderabad/?sort=bestmatch&serveWarrantyCount=true&storeCityId=3686"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers, timeout=15)
soup = BeautifulSoup(response.text, "html.parser")

print("Status:", response.status_code)

# Extract page metadata
for tag in soup.find_all("meta"):
    if tag.get("property") in ["og:title", "og:description"]:
        print(tag.get("property"), ":", tag.get("content", "")[:500])

# Extract visible page text
text = soup.get_text(" ", strip=True)

print("\nVisible text length:", len(text))
print("\nText preview:")
print(text[:2000])

# Find car-price-like strings in the page text
prices = re.findall(
    r"₹\s*[\d,.]+\s*(?:lakh|lakhs|crore|cr)?",
    text,
    re.IGNORECASE
)

print("\nPrice-like values found:", len(prices))
print("First 20:", prices[:20])
