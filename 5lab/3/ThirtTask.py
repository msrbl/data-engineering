import json
from pymongo import MongoClient
from bson import json_util

def read_text_and_insert(file_path, collection):
    records = []
    with open(file_path, "r", encoding="utf-8") as file:
        chunks = file.read().strip().split("=====")
        for chunk in chunks:
            record = {}
            lines = chunk.strip().split("\n")
            for line in lines:
                if "::" in line:
                    key, value = line.split("::")
                    if key in ["salary", "id", "year", "age"]:
                        value = int(value)
                    record[key] = value
            if record:        
                records.append(record)
    
    for record in records:
        try:
            collection.insert_one(record)
        except Exception as e:
            print(f"Пропущена запись с id={record.get('id')}: {e}")

def to_json(collection, file_name):
    data = list(collection.find())
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=json_util.default)

# 1. Удалить документы по предикату: salary < 25 000 || salary > 175 000
def remove_outliers(collection):
    collection.delete_many({"$or": [{"salary": {"$lt": 25000}}, {"salary": {"$gt": 175000}}]})

# 2. Увеличить age всех записей на 1
def increment_age(collection):
    collection.update_many({}, {"$inc": {"age": 1}})

# 3. Поднять заработную плату на 5% для произвольно выбранных профессий
def increase_salary_for_professions(collection, professions):
    collection.update_many(
        {"profession": {"$in": professions}},
        {"$mul": {"salary": 1.05}}
    )

# 4. Поднять заработную плату на 7% для произвольно выбранных городов
def increase_salary_for_cities(collection, cities):
    collection.update_many(
        {"city": {"$in": cities}},
        {"$mul": {"salary": 1.07}}
    )

# 5. Поднять заработную плату на 10% для сложного предиката
def increase_salary_complex_predicate(collection, city, professions, age_range):
    collection.update_many(
        {
            "city": city,
            "profession": {"$in": professions},
            "age": {"$gte": age_range[0], "$lte": age_range[1]}
        },
        {"$mul": {"salary": 1.10}}
    )

# 6. Удалить записи по произвольному предикату
def remove_by_predicate(collection, predicate):
    collection.delete_many(predicate)

if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")
    db = client["data_engineering"]
    collection = db["jobs"]

    read_text_and_insert("task_3_item.text", collection)
    
    to_json(collection, "original_data.json")

    remove_outliers(collection)
    increment_age(collection)
    increase_salary_for_professions(collection, ["Продавец", "Бухгалтер"])
    increase_salary_for_cities(collection, ["Фигерас", "Барселона"])
    increase_salary_complex_predicate(
        collection,
        "Фигерас",
        ["Учитель", "Водитель"],
        [25, 40]
    )
    remove_by_predicate(collection, {"city": "Фигерас", "salary": {"$lt": 50000}})
    
    to_json(collection, "processed_data.json")