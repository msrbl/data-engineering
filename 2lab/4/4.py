import json
import pickle as pkl


def doing_staff(pkl_path, json_path, updated_pkl_path):
    with open(pkl_path, 'rb') as pkl_file:
        data = pkl.load(pkl_file)

    with open(json_path, 'r', encoding='utf-8') as json_file:
        update_data = json.load(json_file)
    
    data_dict = {item["name"]: item for item in data} #Преобразование в словарь для удобства обращения к продуктам

    for item in update_data:
        product_name = item["name"]
        method = item["method"]
        param = item["param"]

        if product_name in data_dict:
            current_price = data_dict[product_name]["price"]

            if method == "add":
                new_price = current_price + param
            elif method == "sub":
                new_price = current_price - param
            elif method == "percent+":
                new_price = current_price + current_price * param
            elif method == "percent-":
                new_price = current_price - current_price * param
            
            data_dict[product_name]["price"] = new_price
    
    data_list = list(data_dict.values()) #Формирования списка из получившегося словаря для сериализации в pickle формат

    with open(updated_pkl_path, "wb") as updated_pkl_file:
        pkl.dump(data_list, updated_pkl_file)


pkl_path = "fourth_task_products.json"
json_path = "fourth_task_updates.json"
updated_pkl_path = "result_pkl.json"

doing_staff(pkl_path, json_path, updated_pkl_path)