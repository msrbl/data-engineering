from bs4 import BeautifulSoup
import requests
import json
import statistics
from collections import Counter

def parse_html(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    product_items = soup.find_all("div", class_="prodCont")
    products = []

    for item in product_items:
        product_id = item.find("div", class_="productItem")["data-id"]

        title_tag = item.find("div", class_="title")
        title = title_tag.text.strip() if title_tag else None

        link_tag = item.find("a")
        link = link_tag["href"] if link_tag else None

        image_tag = item.find("div", class_="picture")["style"]
        if "background-image" in image_tag:
            image_url = image_tag.split("url(")[1].split(")")[0].strip("'\"")

        covering_tag = item.find("div", class_="covering")
        covering = covering_tag.text.split(':')[1].strip() if covering_tag else None
        
        current_color_tag = item.find("div", class_="color")
        color = current_color_tag.text.split(':')[1].strip() if current_color_tag else None

        colors = []
        color_tags = item.find_all("div", class_="chooseColor")
        for color_tag in color_tags:
            color_name = color_tag.find("img")["title"]
            color_image = color_tag.find("img")["src"]
            colors.append({"Название": color_name, "Изображение": color_image})

        price_tag = item.find("span", class_="price")
        price = int(price_tag.text.replace(' ', '').replace('₽', '').strip()) if price_tag else None

        products.append({
            "id": product_id,
            "Название": title,
            "Гиперссылка": link,
            "Изображение": image_url,
            "Цена": price,
            "Покрытие": covering,
            "Текущий цвет": color,
            "Цвета": colors,
        })

    return products

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

url = 'https://alberodoors.com/?ysclid=m49ptigjgi930891012'

data = []
data += parse_html(url)

sorted_data = sorted(data, key=lambda x: x.get('Цена'))

filtered_data = [item for item in data for color in item.get('Цвета') if "Платина" in color.get('Название')]

values = [item.get('Цена') for item in data]

price_stats = {
    "sum": sum(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.mean(values),
    "median": statistics.median(values),
    "std_dev": statistics.stdev(values)
}

covering_frequency = Counter(item['Покрытие'] for item in data)

values_statistics = {
    "Статистика цены": price_stats,
    "Частота покрытий": covering_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_price_data.json')
save_to_json(filtered_data, 'filtered_by_having_Platina_color_data.json')
save_to_json(values_statistics, 'statistics.json')