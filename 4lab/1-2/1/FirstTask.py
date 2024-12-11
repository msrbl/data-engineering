import sqlite3
import pandas as pd
import os

import json

parent_div = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(parent_div, "item.json")

'''# Создание таблицы на основе входного файла
def create_table_from_file(input_file, db_name):
    # Читаем данные из файла
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Конвертируем JSON в DataFrame
    df = pd.DataFrame(data)
    table_name = "data"

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Определяем типы данных для столбцов
    column_definitions = []
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == "int64":
            sql_type = "INTEGER"
        elif dtype == "float64":
            sql_type = "REAL"
        else:
            sql_type = "TEXT"
        column_definitions.append(f"{col} {sql_type}")
    columns = ", ".join(column_definitions)
    cursor.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({columns})")

    # Заполняем таблицу данными
    df.to_sql(table_name, conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()'''

def create_table_from_file(input_file, db_name):
    # Читаем данные из файла
    with open(input_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Конвертируем JSON в DataFrame
    df = pd.DataFrame(data)
    table_name = "data"

    with sqlite3.connect(db_name) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

def to_json(data, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def ordered_query(db_name, output_file, num_records, numeric_field):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = f"SELECT * FROM data ORDER BY {numeric_field} LIMIT {num_records}"
    rows = cursor.execute(query).fetchall()

    column_names = [description[0] for description in cursor.description]

    formatted_data = [dict(zip(column_names, row)) for row in rows]

    conn.close()

    to_json(formatted_data, output_file)

def numeric_stats(db_name, numeric_field):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    query = f"SELECT SUM({numeric_field}), MIN({numeric_field}), MAX({numeric_field}), AVG({numeric_field}) FROM data"
    stats = cursor.execute(query).fetchone()
    conn.close()
    return {"sum": stats[0], "min": stats[1], "max": stats[2], "avg": stats[3]}

def system_frequency(db_name, field):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    query = f"SELECT {field}, COUNT(*) FROM data GROUP BY {field}"
    frequencies = cursor.execute(query).fetchall()
    conn.close()
    return {row[0]: row[1] for row in frequencies}

def filtered_ordered_query(db_name, output_file, num_records, numeric_field, predicate):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    query = f"SELECT * FROM data WHERE {predicate} ORDER BY {numeric_field} LIMIT {num_records}"
    rows = cursor.execute(query).fetchall()

    column_names = [description[0] for description in cursor.description]

    formatted_data = [dict(zip(column_names, row)) for row in rows]

    conn.close()

    to_json(formatted_data, output_file)

if __name__ == "__main__":
    create_table_from_file(input_file, "main_data.db")

    ordered_query("main_data.db", "34_sorted_by_time_on_game.json", 34, "time_on_game")

    stats = numeric_stats("main_data.db", "min_rating")
    frequencies = system_frequency("main_data.db", "system")

    result = {
        "min_rating_stats": stats,
        "system_frequencies": frequencies
    }
    to_json(result, "stats_freqs.json")

    filtered_ordered_query("main_data.db", "34_sorted_by_min_rating_filtered_tours_count.json", 34, "min_rating", "tours_count > 10")