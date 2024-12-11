import csv
import json
import pickle as pkl
import msgpack as msg
import math
import os

def doing_staff(input_csv_path, json_output_path, csv_output_path, msg_output_path, pkl_output_path):
    unique_players = []
    count_left, count_right = 0, 0
    count_joined_after_2012 = 0
    num_records = 0

    metrics = {
        "Age": [],
        "Overall rating": [],
        "Height": [],
        "Weight": [],
        "Wage": [],
        "Total stats": []
    }

    # Чтение данных из CSV файла
    with open(input_csv_path, mode="r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            num_records += 1
            
            # Уникальные имена игроков
            unique_players.append(row["player_name"])
            
            # Сбор данных по метрикам
            metrics["Age"].append(int(row["Age"]))
            metrics["Overall rating"].append(int(row["Overall rating"][:2]))
            metrics["Height"].append(int(row["Height"]))
            metrics["Weight"].append(int(row["Weight"]))
            wage = int(row["Wage"].replace("€", "").replace("K", "000").replace("M", "000000"))
            metrics["Wage"].append(wage)
            metrics["Total stats"].append(int(row["Total stats"]))
            
            # Подсчет количества левоногих и правоногих
            if row["foot"] == "Left":
                count_left += 1
            elif row["foot"] == "Right":
                count_right += 1

            # Год в "Joined" > 2012
            if row["Joined"]:
                joined_year = int(row["Joined"][-4:])
                if joined_year > 2012:
                    count_joined_after_2012 += 1


    def calculate_statistics(data):
        avg = sum(data) / len(data)
        std_dev = math.sqrt(sum((x - avg) ** 2 for x in data) / len(data))
        return {
            "sum": sum(data),
            "average": round(avg, 3),
            "max": max(data),
            "min": min(data),
            "std_dev": round(std_dev, 3)
        }

    metrics_stats = {metric: calculate_statistics(values) for metric, values in metrics.items()}

    # Формирование словаря результатов
    result = {
        "unique_player_count": len(unique_players),
        "left_foot_count": count_left,
        "right_foot_count": count_right,
        "joined_after_2012_count": count_joined_after_2012,
        "metrics": metrics_stats
    }

    #JSON
    with open(json_output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(result, json_file, indent=4)

    #Message Pack
    with open(msg_output_path, 'wb') as msg_file:
        msg.pack(result, msg_file)

    #CSV
    with open(csv_output_path, "w", encoding="utf-8", newline="") as csv_file:
        writer =csv.writer(csv_file)
        writer.writerow(["Metric", "Sum", "Average", "Max", "Min", "Std Dev"])
        for metric, stats in metrics_stats.items():
            writer.writerow([metric, stats["sum"], stats["average"], stats["max"], stats["min"], stats["std_dev"]])

    #Pickle
    with open(pkl_output_path, "wb") as pkl_file:
        pkl.dump(result, pkl_file)

    json_size = os.path.getsize(json_output_path)
    msg_size = os.path.getsize(msg_output_path)
    csv_size = os.path.getsize(csv_output_path)
    pkl_size = os.path.getsize(pkl_output_path)

    with open("comparison_sizes.txt", "w", encoding="utf-8") as txt_file:
        txt_file.write("Сравнение размеров файлов разных форматов:\n")
        txt_file.write(f"Файл формата JSON весит: {json_size} байт\n")
        txt_file.write(f"Файл формата Message Pack весит: {msg_size} байт\n")
        txt_file.write(f"Файл формата CSV весит: {csv_size} байт\n")
        txt_file.write(f"Файл формата Pickle весит: {pkl_size} байт\n")

input_csv_path = "fifa_players.csv"
json_output_path = "json_result.json"
csv_output_path = "csv_result.csv"
msg_output_path = "msg_result.msg"
pkl_output_path = "pkl_result.pkl"

doing_staff(input_csv_path, json_output_path, csv_output_path, msg_output_path, pkl_output_path)