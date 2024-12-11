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

    data = {}
    for child in root:
        data[child.tag] = child.text.strip() if child.text else None
    return data

def save_to_json(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

data = []
for file_name in os.listdir(xml_dir):
    if file_name.endswith('.xml'):
        file_path = os.path.join(xml_dir, file_name)
        data.append(parse_xml(file_path))

sorted_data = sorted(data, key=lambda x: float(x['distance'].split()[0]) if x['distance'].split('.')[0].isdigit() else 0)

filtered_data = [item for item in data if item.get('constellation') == 'Козерог']

radius_values = [float(item['radius']) for item in data if 'radius' in item and item['radius'].isdigit()]
radius_stats = {
    "sum": sum(radius_values),
    "min": min(radius_values),
    "max": max(radius_values),
    "mean": statistics.mean(radius_values),
    "median": statistics.median(radius_values),
    "std_dev": statistics.stdev(radius_values)
}

constellation_values = [item['constellation'] for item in data if 'constellation' in item]
constellation_frequency = Counter(constellation_values)

xml_statistics = {
    "Статистика по радиусу": radius_stats,
    "Частота созвездий": constellation_frequency
}

save_to_json(data, 'parsed_data.json')
save_to_json(sorted_data, 'sorted_by_distance_data.json')
save_to_json(filtered_data, 'filtered_by_constellation_is_Kozerog_data.json')
save_to_json(xml_statistics, 'statistics.json')