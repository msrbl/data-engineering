import json
import msgpack
import os


def doing_staff(file_path, json_file_path, msg_file_path):
    with open(file_path, 'r', encoding='utf-8') as file_load:
        data = json.load(file_load)

    aggregated_data = {}

    for item in data:
        product = item['name']
        price = item['price']

        if product not in aggregated_data:
            aggregated_data[product] = {"prices": []}
        
        aggregated_data[product]["prices"].append(price)

    for product, values in aggregated_data.items():
        prices = values["prices"]
        aggregated_data[product] = {
            "average_price": round(sum(prices) / len(prices), 6),
            "max_price": max(prices),
            "min_price": min(prices)
        }

    with open(json_file_path, 'w', encoding="utf-8") as file_json:
        json.dump(aggregated_data, file_json, indent=4, ensure_ascii=False)

    with open(msg_file_path, 'wb') as file_msg:
        msgpack.pack(aggregated_data, file_msg)

    json_size = os.path.getsize(json_file_path)
    msg_size = os.path.getsize(msg_file_path)
    diff = json_size / msg_size

    result = f'''Размер файла в формате json: {json_size} байт
Размер файла в формате msg: {msg_size} байт

Разница json к msg: {diff:.2f}
'''

    with open("comparison_sizes.txt", "w", encoding="utf-8") as f:
        f = f.write(result)


json_path = "third_task.json"
result_json_path = "result_json.json"
result_msg_path = "result_msg.msg"

doing_staff(json_path, result_json_path, result_msg_path)

