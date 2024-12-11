import pandas as pd
import sqlite3
import os

import msgpack
import json

parent_div = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(parent_div, "subitem.msgpack")

def create_table_from_file(msg_file, db_name, table_name):
    with open(msg_file, "rb") as file:
        data = msgpack.unpackb(file.read(), raw=False)

    df = pd.DataFrame(data)

    with sqlite3.connect(db_name) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)

def get_by_TOG_and_prize(conn, min_time_on_game, min_prize):
    query = """
    SELECT a.name, a.time_on_game, b.place, b.prise
    FROM data a
    JOIN add_data b 
    ON a.name = b.name
    WHERE a.time_on_game > ? AND b.prise > ?
    """
    return conn.execute(query, (min_time_on_game, min_prize)).fetchall()

def get_by_system_and_prize(conn, system, min_prize):
    query = """
    SELECT a.name, a.system, b.place, b.prise 
    FROM data a
    JOIN add_data b
    ON a.name = b.name
    WHERE a.system = ? AND b.prise > ?
    """
    return conn.execute(query, (system, min_prize)).fetchall()

def get_avg_prize_by_city(conn):
    query = """
    SELECT a.city, AVG(b.prise) AS avg_prize 
    FROM data a
    JOIN add_data b
    ON a.name = b.name
    GROUP BY a.city
    ORDER BY avg_prize DESC
    """
    return conn.execute(query).fetchall()

def save_results_to_json(results, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main_database_name = "main_data.db"
    add_table_name = "add_data"
    create_table_from_file(input_file, main_database_name, add_table_name)

    with sqlite3.connect(main_database_name) as conn:
        high_prize_results = get_by_TOG_and_prize(conn, 10, 10000000)
        save_results_to_json(high_prize_results, "high_prize_tournaments.json")

        swiss_system_results = get_by_system_and_prize(conn, "Swiss", 5000000)
        save_results_to_json(swiss_system_results, "swiss_system_tournaments.json")

        avg_prize_results = get_avg_prize_by_city(conn)
        save_results_to_json(avg_prize_results, "avg_prize_by_city.json")
