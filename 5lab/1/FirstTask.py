import pickle
from pymongo import MongoClient
import json
from bson import json_util

def create_collection_from_pkl(input_file):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["data_engineering"]
    collection = db["jobs"]

    collection.create_index("id", unique=True)

    with open(input_file, "rb") as f:
        data = pickle.load(f)

    for record in data:
        try:
            collection.insert_one(record)
        except Exception as e:
            print(f"Пропущена запись с id={record['id']}: {e}")

    return collection

def to_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=json_util.default)

# 10 записей, отсортированных по убыванию по salary
def first_query(collection):
    query1 = collection.find().sort("salary", -1).limit(10)
    
    to_json(query1.to_list(), "salary_down_sort.json")

# 15 записей, отфильтрованных по age < 30, отсортированные по убыванию по salary
def second_query(collection):
    query2 = collection.find({"age": {"$lt": 30}}).sort("salary", -1).limit(15)
    
    to_json(query2.to_list(), "age_below_30_salary_down_sort.json")

# 10 записей, отфильтрованных по городу и профессиям, отсортированных по возрасту по возрастанию
def third_query(collection):
    query3 = collection.find(
        {"city": "Фигерас", "job": {"$in": ["Учитель", "Продавец", "Водитель"]}}
    ).sort("age", 1).limit(10)
    
    to_json(query3.to_list(), "figeras_three_proffesions_age_up_sort.json")

# Количество записей с 25 < age < 50, year in [2019, 2020, 2021, 2022] && 50000 < salary < 75000 && 125000 < salary < 150000
def forth_query(collection):
    query4_filter = {
        "age": {"$gte": 25, "$lte": 50},
        "year": {"$in": [2019, 2020, 2021, 2022]},
        "$or": [
            {"salary": {"$gt": 50000, "$lte": 75000}},
            {"salary": {"$gt": 125000, "$lt": 150000}},
        ],
    }
    query4_count = collection.count_documents(query4_filter)
    
    to_json({"count": query4_count}, "counted_filtered_records.json")

if __name__ == "__main__":
    collection = create_collection_from_pkl("task_1_item.pkl")

    first_query(collection)
    second_query(collection)
    third_query(collection)
    forth_query(collection)