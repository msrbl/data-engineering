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
        product_items = soup.find_all('div', class_='product-item')
        products = []

        for item in product_items:
            product = {}
            product['id'] = int(item.find('a', class_='add-to-favorite')['data-id'])
            product['Название'] = item.find('span').text.strip()
            product['Цена'] = int(item.find('price').text.replace('₽', '').replace(' ', '').strip())
            product['Бонусы'] = int(item.find('strong').text.split()[2])
            
            # Extracting specifications
            specs = {}
            for li in item.find_all('li'):
                spec_type = li.get('type')
                spec_value = li.text.strip()
                specs[spec_type] = spec_value
            product['specifications'] = specs
            
            products.append(product)
    
        return products

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = []
for file_name in os.listdir(html_dir):
    if file_name.endswith('.html'):
        file_path = os.path.join(html_dir, file_name)
        data += parse_html(file_path)

sorted_data = sorted(data, key=lambda x: x['Цена'])

filtered_data = [item for item in data if 'ram' in item.get('specifications')]

values = [item['Цена'] for item in data]

price_stats = {
    "sum": sum(values),
    "min": min(values),
    "max": max(values),
    "mean": statistics.mean(values),
    "median": statistics.median(values),
    "std_dev": statistics.stdev(values)
}

specifications_frequency = Counter(key for item in data for key in item['specifications'].keys())

values_statistics = {
    "Статистика цены": price_stats,
    "Частота характеристик": specifications_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_price_data.json')
save_to_json(filtered_data, 'filtered_by_having_ram_data.json')
save_to_json(values_statistics, 'statistics.json')