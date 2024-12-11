from bs4 import BeautifulSoup
import requests
import json
import statistics
from collections import Counter

def parse_html(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    product_items = soup.find_all("div", class_="content")
    products = []

    for product in product_items:
        product_id = product["data-id"]

        title_tag = product.find("div", class_="heading")
        title = title_tag.text.strip() if title_tag else None

        detail_tag = product.find("div", class_="detail-detail_text")
        detail = detail_tag.text.strip() if detail_tag else None

        price_tag = product.find("span", class_="price")
        price = int(price_tag.text.replace(' ', '').replace('₽', '').strip()) if price_tag else None

        image_tag = product.find("div", class_="bigPic").find("img")["src"]
        image_url = image_tag.strip() if image_tag else None

        colors = []
        color_list = product.find("div", class_="colorList")
        color_tags = color_list.find_all("div", class_="item")
        for color_tag in color_tags:
            color_name = color_tag["title"]
            color_image = color_tag["style"].split("url(")[1].split(")")[0].strip("'\"")
            colors.append({"Название": color_name, "Изображение": color_image})

        characteristics = []
        char_tag = product.find_all("div", class_="characteristic")
        for char in char_tag:
            char_title_tag = char.find("div", class_="characteristicTitle")
            char_title = char_title_tag.text.replace(':', '').strip() if char_title_tag else None
            char_text_tag = char.find("div", class_="characteristicText")
            char_text = char_text_tag.text.replace(':', '').strip() if char_text_tag else None
            if char_title_tag and char_text_tag:
                characteristics.append({char_title: char_text})

        products.append({
            "id": product_id,
            "Название": title,
            "Описание": detail,
            "Гиперссыкла": url,
            "Изображение": image_url,
            "Цена": price,
            "Цвета": colors,
            "Характеристики": characteristics
        })

    return products

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

url_base = 'https://alberodoors.com/'
data = []

with open("Catalog/parsed_data.json", "r", encoding="utf-8") as json_file:
    urls = json.load(json_file)
    for product in urls[:5]:
        url = product.get('Гиперссылка')
        data += parse_html(url_base + url)

sorted_data = sorted(data, key=lambda x: x.get('Цена'))

filtered_data = [item for item in data for char in item.get('Характеристики') if char.get('Цвет') == "Белый" and item.get('Цена') < 10000]

values = [item.get('Цена') for item in data]

price_stats = {
    "sum": sum(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.mean(values),
    "median": statistics.median(values),
    "std_dev": statistics.stdev(values)
}

colors_frequency = Counter(color['Название'] for item in data for color in item.get('Цвета'))

values_statistics = {
    "Статистика цены": price_stats,
    "Частота цветов": colors_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_price_data.json')
save_to_json(filtered_data, 'filtered_by_white_color_and_price_under_10000_data.json')
save_to_json(values_statistics, 'statistics.json')