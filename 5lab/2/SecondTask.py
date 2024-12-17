import pandas as pd
from pymongo import MongoClient
from bson import json_util
import json

def add_csv_to_collection(csv_file, collection):
    df = pd.read_csv(csv_file, delimiter=";")
    records = df.to_dict(orient="records")

    for record in records:
        try:
            collection.insert_one(record)
        except Exception as e:
            print(f"Пропущена запись с id={record.get('id')}: {e}")

def to_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, default=json_util.default)

# Вывод минимальной, средней, максимальной salary
def salary_stats(collection):
    query = [
        {"$group": {
            "_id": None,
            "min_salary": {"$min": "$salary"},
            "avg_salary": {"$avg": "$salary"},
            "max_salary": {"$max": "$salary"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "salary_stats.json")

# 2. Количество данных по представленным профессиям
def job_count(collection):
    query = [
        {"$group": {
            "_id": "$job",
            "count": {"$sum": 1}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "profession_count.json")

# 3. Минимальная, средняя, максимальная salary по городу
def salary_by_city(collection):
    query = [
        {"$group": {
            "_id": "$city",
            "min_salary": {"$min": "$salary"},
            "avg_salary": {"$avg": "$salary"},
            "max_salary": {"$max": "$salary"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "salary_by_city.json")

# 4. Минимальная, средняя, максимальная salary по профессии
def salary_by_job(collection):
    query = [
        {"$group": {
            "_id": "$job",
            "min_salary": {"$min": "$salary"},
            "avg_salary": {"$avg": "$salary"},
            "max_salary": {"$max": "$salary"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "salary_by_job.json")

# 5. Минимальный, средний, максимальный возраст по городу
def age_by_city(collection):
    query = [
        {"$group": {
            "_id": "$city",
            "min_age": {"$min": "$age"},
            "avg_age": {"$avg": "$age"},
            "max_age": {"$max": "$age"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "age_by_city.json")

# 6. Минимальный, средний, максимальный возраст по профессии
def age_by_job(collection):
    query = [
        {"$group": {
            "_id": "$job",
            "min_age": {"$min": "$age"},
            "avg_age": {"$avg": "$age"},
            "max_age": {"$max": "$age"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "age_by_job.json")

# 7. Максимальная salary при минимальном возрасте
def max_salary_min_age(collection):
    query = [
        {"$sort": {"age": 1, "salary": -1}},
        {"$limit": 1}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "max_salary_min_age.json")

# 8. Минимальная salary при максимальном возрасте
def min_salary_max_age(collection):
    query = [
        {"$sort": {"age": -1, "salary": 1}},
        {"$limit": 1}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "min_salary_max_age.json")

# 9. Возрастные статистики по городу с salary > 50 000, сортировка по avg_age
def age_stats_high_salary(collection):
    query = [
        {"$match": {"salary": {"$gt": 50000}}},
        {"$group": {
            "_id": "$city",
            "min_age": {"$min": "$age"},
            "avg_age": {"$avg": "$age"},
            "max_age": {"$max": "$age"}
        }},
        {"$sort": {"avg_age": -1}}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "age_stats_high_salary.json")

# 10. Salary в диапазонах возраста по городу и профессии
def salary_in_ranges(collection):
    query = [
        {"$match": {"$or": [
            {"age": {"$gt": 18, "$lt": 25}},
            {"age": {"$gt": 50, "$lt": 65}}
        ]}},
        {"$group": {
            "_id": {"city": "$city", "profession": "$profession"},
            "min_salary": {"$min": "$salary"},
            "avg_salary": {"$avg": "$salary"},
            "max_salary": {"$max": "$salary"}
        }}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "salary_in_ranges.json")

# 11. Произвольный запрос
def custom_query(collection):
    query = [
        {"$match": {"city": "Фигерас", "salary": {"$gt": 60000}}},
        {"$group": {
            "_id": "$profession",
            "total_salary": {"$sum": "$salary"}
        }},
        {"$sort": {"total_salary": -1}}
    ]
    result = list(collection.aggregate(query))
    to_json(result, "custom_query.json")

if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")
    db = client["data_engineering"]
    collection = db["jobs"]

    add_csv_to_collection("task_2_item.csv", collection)

    salary_stats(collection)
    job_count(collection)
    salary_by_city(collection)
    salary_by_job(collection)
    age_by_city(collection)
    age_by_job(collection)
    max_salary_min_age(collection)
    min_salary_max_age(collection)
    age_stats_high_salary(collection)
    salary_in_ranges(collection)
    custom_query(collection)