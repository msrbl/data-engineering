import xml.etree.ElementTree as ET
import json
import os
import statistics
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
xml_dir = os.path.join(current_dir, 'xml')

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    items = []
    for clothing in root.findall('clothing'):
        item = {child.tag: child.text.strip() if child.text else None for child in clothing}
        items.append(item)
    return items

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = []
for file_name in os.listdir(xml_dir):
    if file_name.endswith('.xml'):
        file_path = os.path.join(xml_dir, file_name)
        data += parse_xml(file_path)

sorted_data = sorted(data, key=lambda x: x.get('price', 0))

filtered_data = [item for item in data if item.get('category') == 'Shirt']

for item in data:
    for key in ['price', 'rating', 'reviews']:
        if key in item and item[key] is not None:
            item[key] = float(item[key]) if '.' in item[key] else int(item[key])

ratings_values = [item['rating'] for item in data if 'rating' in item]
rating_stats = {
    "sum": sum(ratings_values),
    "min": min(ratings_values),
    "max": max(ratings_values),
    "mean": statistics.mean(ratings_values),
    "median": statistics.median(ratings_values),
    "std_dev": statistics.stdev(ratings_values)
}

colors_values = [item.get('color') for item in data if item.get('color')]
color_frequency = Counter(colors_values)

xml_statistics = {
    "Статистика по рейтингу": rating_stats,
    "Частота цветов": color_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_price_data.json')
save_to_json(filtered_data, 'filtered_by_category_is_shirt_data.json')
save_to_json(xml_statistics, 'statistics.json')