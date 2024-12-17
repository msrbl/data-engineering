import pandas as pd
import os
import json
import msgpack
from pymongo import MongoClient

def convert_numeric_fields(record, numeric_fields):
    for field in numeric_fields:
        if field in record:
            try:
                if record[field].isnumeric():
                    record[field] = int(record[field])
                else:
                    record[field] = float(record[field])
            except ValueError and AttributeError:
                continue
    return record

def load_csv_to_mongo(csv_file, collection):
    df = pd.read_csv(csv_file, delimiter=";")
    records = df.to_dict(orient="records")
    numeric_fields = ["duration_ms", "year", "tempo", "energy", "key", "loudness"]

    for record in records:
        record = convert_numeric_fields(record, numeric_fields)

    collection.insert_many(records)

def load_msgpack_to_mongo(msgpack_file, collection):
    with open(msgpack_file, 'rb') as file:
        data = msgpack.unpack(file, raw=False)
    df = pd.DataFrame(data)
    records = df.to_dict(orient="records")
    numeric_fields = [
        "duration_ms", "year", "tempo", "mode",
        "speechiness", "acousticness", "instrumentalness"
    ]

    for record in records:
        record = convert_numeric_fields(record, numeric_fields)

        query = {"artist": record["artist"], "song": record["song"]}
        existing_record = collection.find_one(query)

        if existing_record:
            update_fields = {k: v for k, v in record.items() if k not in existing_record}
            if update_fields:
                collection.update_one(query, {"$set": update_fields})
        else:
            collection.insert_one(record)

def to_json(results, folder_name, query_name):
    os.makedirs(folder_name, exist_ok=True)
    file_path = os.path.join(folder_name, f"{query_name}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

# 5 запросов на выборку
def selection_queries(collection):
    folder_name = "selection"

    # 1. Все записи с жанром "pop"
    results = list(collection.find({"genre": {"$regex": ".*pop.*", "$options": "i"}}, {"_id": 0}))
    to_json(results, folder_name, "pop_genre_songs")

    # 2. Песни после 2015 года
    results = list(collection.find({"year": {"$gte": 2015}}, {"_id": 0}))
    to_json(results, folder_name, "recent_songs")

    # 3. Песни, где энергия выше 0.7
    results = list(collection.find({"energy": {"$gt": 0.7}}, {"_id": 0}))
    to_json(results, folder_name, "high_energy_songs")

    # 4. Записи с длительностью больше 4 минут
    results = list(collection.find({"duration_ms": {"$gt": 240000}}, {"_id": 0}))
    to_json(results, folder_name, "long_songs")

    # 5. Песни с артистом "Ariana Grande"
    results = list(collection.find({"artist": "Ariana Grande"}, {"_id": 0}))
    to_json(results, folder_name, "ariana_grande_songs")

# 5 запросов с агрегацией
def aggregation_queries(collection):
    folder_name = "aggregation"

    # 1. Средняя длительность песен
    results = list(collection.aggregate([{"$group": {"_id": None, "avg_duration": {"$avg": "$duration_ms"}}}]))
    to_json(results, folder_name, "average_duration")

    # 2. Количество песен по каждому жанру
    results = list(collection.aggregate([{"$group": {"_id": "$genre", "count": {"$sum": 1}}}]))
    to_json(results, folder_name, "song_count_by_genre")

    # 3. Максимальная энергия песен по артисту
    results = list(collection.aggregate([{"$group": {"_id": "$artist", "max_energy": {"$max": "$energy"}}}]))
    to_json(results, folder_name, "max_energy_by_artist")

    # 4. Средняя громкость песен по году
    results = list(collection.aggregate([{"$group": {"_id": "$year", "avg_loudness": {"$avg": "$loudness"}}}]))
    to_json(results, folder_name, "average_loudness_by_year")

    # 5. Максимальная громкость песен с энергией > 0.8
    results = list(collection.aggregate([
        {"$match": {"energy": {"$gt": 0.8}}},
        {"$group": {"_id": None, "max_loudness": {"$max": "$loudness"}}}
    ]))
    to_json(results, folder_name, "max_loudness_high_energy")

# 5 запросов на обновление/удаление
def update_and_delete_queries(collection):
    folder_name = "update_delete"

    # 1. Увеличить энергию песен на 10%, где жанр содержит "pop"
    collection.update_many({"genre": {"$regex": ".*pop.*", "$options": "i"}}, {"$mul": {"energy": 1.1}})
    results = list(collection.find({"genre": {"$regex": ".*pop.*", "$options": "i"}}, {"_id": 0}))
    to_json(results, folder_name, "updated_energy_pop_genre")

    # 2. Уменьшить громкость песен на 5 дБ, где громкость ниже -5
    collection.update_many({"loudness": {"$lt": -5}}, {"$inc": {"loudness": -5}})
    results = list(collection.find({"loudness": {"$lt": -5}}, {"_id": 0}))
    to_json(results, folder_name, "updated_loudness_below_minus5")

    # 3. Удалить записи с длительностью менее 2 минут
    deleted_count = collection.delete_many({"duration_ms": {"$lt": 120000}}).deleted_count
    to_json({"deleted_count": deleted_count}, folder_name, "deleted_short_songs")

    # 4. Удалить записи с артистом "Jay Sean"
    deleted_count = collection.delete_many({"artist": "Jay Sean"}).deleted_count
    to_json({"deleted_count": deleted_count}, folder_name, "deleted_jay_sean_songs")

    # 5. Увеличить длительность песен на 1% для записей до 2010 года
    collection.update_many({"year": {"$lt": 2010}}, {"$mul": {"duration_ms": 1.01}})
    results = list(collection.find({"year": {"$lt": 2010}}, {"_id": 0}))
    to_json(results, folder_name, "updated_duration_before_2010")

if __name__ == "__main__":
    client = MongoClient("mongodb://localhost:27017/")
    db = client["data_engineering"]
    collection = db["music_collection"]

    load_csv_to_mongo("music_part_1.csv", collection)
    load_msgpack_to_mongo("music_part_2.msgpack", collection)

    selection_queries(collection)
    aggregation_queries(collection)
    update_and_delete_queries(collection)