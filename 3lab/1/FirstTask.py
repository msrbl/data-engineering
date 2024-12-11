from bs4 import BeautifulSoup
import os
import json
import statistics
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
html_dir = os.path.join(current_dir, 'html')

def parse_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')

        city = soup.find('span', string=lambda x: "Город:" in x).text.split("Город:")[1].strip()
        building = soup.find('h1', class_='title').text.split(":")[1].strip()
        address_parts = soup.find('p', class_='address-p').text.split("Индекс:")
        street = address_parts[0].split("Улица:")[1].strip()
        postal_code = address_parts[1].strip()

        floors = int(soup.find('span', class_='floors').text.split(":")[1].strip())
        year_built = int(soup.find('span', class_='year').text.split()[2].strip())
        parking = soup.find('span', string=lambda x: "Парковка" in x).text.split(":")[1].strip()

        rating = float(soup.find('span', string=lambda x: "Рейтинг" in x).text.split(":")[1].strip())
        views = int(soup.find('span', string=lambda x: "Просмотры" in x).text.split(":")[1].strip())

        return {
            "Город": city,
            "Строение": building,
            "Улица": street,
            "Индекс": postal_code,
            "Этажи": floors,
            "Год": year_built,
            "Парковка": parking,
            "Рейтинг": rating,
            "Просмотры": views
        }

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = []
for file_name in os.listdir(html_dir):
    if file_name.endswith('.html'):
        file_path = os.path.join(html_dir, file_name)
        data.append(parse_html(file_path))

sorted_data = sorted(data, key=lambda x: x['Этажи'])

filtered_data = [item for item in data if item['Парковка'] == "есть"]

values = [item['Просмотры'] for item in data]

views_stats = {
    "sum": sum(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.mean(values),
    "median": statistics.median(values),
    "std_dev": statistics.stdev(values)
}

city_frequency = Counter(item['Город'] for item in data)

values_statistics = {
    "Статистика просмотров": views_stats,
    "Частота городов": city_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_floors_data.json')
save_to_json(filtered_data, 'filtered_by_available_parking_data.json')
save_to_json(values_statistics, 'statistics.json')