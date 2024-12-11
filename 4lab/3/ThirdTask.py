import pandas as pd
import sqlite3
import json
import msgpack

def read_msgpack(file_path):
    with open(file_path, "rb") as file:
        data = msgpack.unpackb(file.read(), raw=False)
    return pd.DataFrame(data)

def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = []
        record = {}
        for line in file:
            line = line.strip()
            if line == "=====" or not line:
                if record:
                    data.append(record)
                    record = {}
            else:
                key, value = line.split("::")
                record[key] = value
        if record:
            data.append(record)
    return pd.DataFrame(data)

def merge_data(msgpack_file, text_file):
    df1 = read_msgpack(msgpack_file)
    df2 = read_text(text_file)
    return pd.concat([df1, df2], ignore_index=True)

def create_table(data, db_name, table_name):
    with sqlite3.connect(db_name) as conn:
        data.to_sql(table_name, conn, if_exists="replace", index=False)

def get_sorted_rows(conn, table_name, numeric_field, limit, output_file):
    query = f"""
    SELECT * FROM {table_name}
    ORDER BY {numeric_field} ASC
    LIMIT ?
    """
    return conn.execute(query, (limit,)).fetchall()

def get_numeric_statistics(conn, table_name, numeric_field):
    query = f"""
    SELECT SUM({numeric_field}) AS sum, MIN({numeric_field}) AS min,
           MAX({numeric_field}) AS max, AVG({numeric_field}) AS avg
    FROM {table_name}
    """
    return conn.execute(query).fetchone()

def get_frequency(conn, table_name, categorical_field):
    query = f"""
    SELECT {categorical_field}, COUNT(*) AS frequency
    FROM {table_name}
    GROUP BY {categorical_field}
    ORDER BY frequency DESC
    """
    return conn.execute(query).fetchall()

def get_filtered_sorted_rows(conn, table_name, numeric_field, predicate, limit, output_file):
    query = f"""
    SELECT * FROM {table_name}
    WHERE {predicate}
    ORDER BY {numeric_field} ASC
    LIMIT ?
    """
    return conn.execute(query, (limit,)).fetchall()

def save_results_to_json(results, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    msgpack_file = "_part_1.msgpack"
    text_file = "_part_2.text"
    database_name = "combined_data.db"
    table_name = "music_data"

    combined_data = merge_data(msgpack_file, text_file)

    create_table(combined_data, database_name, table_name)

    with sqlite3.connect(database_name) as conn:
        sorted_output_file = "sorted_rows.json"
        sorted_rows = get_sorted_rows(conn, table_name, "duration_ms", 34, sorted_output_file)
        save_results_to_json(sorted_rows, sorted_output_file)

        stats_output_file = "stats.json"
        stats = get_numeric_statistics(conn, table_name, "duration_ms")
        save_results_to_json(stats, stats_output_file)

        frequency_output_file = "frequency.json"
        frequency = get_frequency(conn, table_name, "genre")
        save_results_to_json(frequency, frequency_output_file)

        filtered_sorted_output_file = "filtered_sorted_rows.json"
        predicate = "tempo > 100 AND instrumentalness > 0.5"
        filtered_sorted_rows = get_filtered_sorted_rows(conn, table_name, "duration_ms", predicate, 34, filtered_sorted_output_file)
        save_results_to_json(filtered_sorted_rows, filtered_sorted_output_file)