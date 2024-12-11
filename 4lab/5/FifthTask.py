'''

Музыкальная база данных, включающая информацию о песнях и их характеристиках.
Таблицы:
- songs: данные о песнях (исполнитель, песня, продолжительность, год и др.).
- genres: жанры песен.
- features: дополнительные характеристики песен (акустичность, инструментальность и др.).

'''

import pandas as pd
import sqlite3
import json
import msgpack

CREATE_SONGS_TABLE = """
CREATE TABLE IF NOT EXISTS songs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    artist TEXT,
    song TEXT,
    duration_ms INTEGER,
    year INTEGER,
    tempo REAL
);
"""

CREATE_GENRES_TABLE = """
CREATE TABLE IF NOT EXISTS genres (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    genre TEXT,
    FOREIGN KEY (song_id) REFERENCES songs (id)
);
"""

CREATE_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    song_id INTEGER,
    energy REAL,
    loudness REAL,
    acousticness REAL,
    instrumentalness REAL,
    FOREIGN KEY (song_id) REFERENCES songs (id)
);
"""

def initialize_database(db_name):
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        cursor.execute(CREATE_SONGS_TABLE)
        cursor.execute(CREATE_GENRES_TABLE)
        cursor.execute(CREATE_FEATURES_TABLE)

def load_csv_data_to_db(csv_file, db_name):
    df = pd.read_csv(csv_file, delimiter=';')
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO songs (artist, song, duration_ms, year, tempo) VALUES (?, ?, ?, ?, ?)",
                (row['artist'], row['song'], row['duration_ms'], row['year'], row['tempo'])
            )
            song_id = cursor.lastrowid
            genres = row['genre'].split(', ')
            for genre in genres:
                cursor.execute("INSERT INTO genres (song_id, genre) VALUES (?, ?)", (song_id, genre))
            cursor.execute(
                "INSERT INTO features (song_id, energy, loudness) VALUES (?, ?, ?)",
                (song_id, row['energy'], row['loudness'])
            )

def load_msgpack_data_to_db(msgpack_file, db_name):
    with open(msgpack_file, 'rb') as file:
        data = msgpack.unpack(file, raw=False)
    df = pd.DataFrame(data)
    with sqlite3.connect(db_name) as conn:
        cursor = conn.cursor()
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO songs (artist, song, duration_ms, year, tempo) VALUES (?, ?, ?, ?, ?)",
                (row['artist'], row['song'], row['duration_ms'], row['year'], row['tempo'])
            )
            song_id = cursor.lastrowid
            genres = row['genre'].split(', ')
            for genre in genres:
                cursor.execute("INSERT INTO genres (song_id, genre) VALUES (?, ?)", (song_id, genre))
            cursor.execute(
                "INSERT INTO features (song_id, acousticness, instrumentalness) VALUES (?, ?, ?)",
                (song_id, row['acousticness'], row['instrumentalness'])
            )

def save_to_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

def execute_queries(db_name):
    with sqlite3.connect(db_name) as conn:
        # Выборка с условием + сортировка + ограничение количества
        query1 = """
        SELECT artist, song, year, tempo FROM songs
        WHERE year > 2010
        ORDER BY tempo DESC
        LIMIT 15
        """
        result1 = conn.execute(query1).fetchall()
        save_to_json('query1.json', result1)

        # Подсчет объектов по условию и функции агрегации
        query2 = """
        SELECT genre, COUNT(*) AS count, MIN(tempo), MAX(tempo), AVG(tempo)
        FROM genres
        JOIN songs ON genres.song_id = songs.id
        GROUP BY genre
        """
        result2 = conn.execute(query2).fetchall()
        save_to_json('query2.json', result2)

        # Группировка по исполнителям и средние значения характеристик
        query3 = """
        SELECT artist, AVG(energy) AS avg_energy, AVG(loudness) AS avg_loudness
        FROM features
        JOIN songs ON features.song_id = songs.id
        GROUP BY artist
        """
        result3 = conn.execute(query3).fetchall()
        save_to_json('query3.json', result3)

        # Группировка по годам и подсчет песен
        query4 = """
        SELECT year, COUNT(*) AS song_count
        FROM songs
        GROUP BY year
        ORDER BY year DESC
        """
        result4 = conn.execute(query4).fetchall()
        save_to_json('query4.json', result4)

        # Подсчет среднего темпа и максимального уровня энергии по жанрам
        query5 = """
        SELECT genre, AVG(tempo) AS avg_tempo, MAX(energy) AS max_energy
        FROM genres
        JOIN songs ON genres.song_id = songs.id
        JOIN features ON songs.id = features.song_id
        GROUP BY genre
        """
        result5 = conn.execute(query5).fetchall()
        save_to_json('query5.json', result5)

        # Выборка песен с акустичностью выше среднего
        query6 = """
        SELECT songs.artist, songs.song, features.acousticness
        FROM songs
        JOIN features ON songs.id = features.song_id
        WHERE features.acousticness > (SELECT AVG(acousticness) FROM features)
        ORDER BY features.acousticness DESC
        """
        result6 = conn.execute(query6).fetchall()
        save_to_json('query6.json', result6)

        # Поиск песен, продолжительность которых превышает 5 минут
        query7 = """
        SELECT artist, song, duration_ms
        FROM songs
        WHERE duration_ms > 300000
        ORDER BY duration_ms DESC
        """
        result7 = conn.execute(query7).fetchall()
        save_to_json('query7.json', result7)

if __name__ == "__main__":
    database_name = "music_data.db"
    initialize_database(database_name)

    csv_file = "music_part_1.csv"
    msgpack_file = "music_part_2.msgpack"

    load_csv_data_to_db(csv_file, database_name)
    load_msgpack_data_to_db(msgpack_file, database_name)

    execute_queries(database_name)
