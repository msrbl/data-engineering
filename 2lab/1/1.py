import numpy as np
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.float):
            return float(obj)
        return super(NpEncoder, self).default(obj)

def doing_staff():
    matrix = np.load('first_task.npy')

    matrix_sum = np.sum(matrix)
    matrix_average = round(np.mean(matrix), 6)

    MD_sum = sum(matrix.diagonal())
    MD_avg = round(MD_sum / len(matrix), 6)

    SD_sum = np.sum(np.fliplr(matrix).diagonal())
    SD_avg = round(SD_sum / len(matrix), 6)

    max_value = np.max(matrix)
    min_value = np.min(matrix)

    normalized_matrix = (matrix - min_value) / (max_value - min_value)

    np.save("normalized_matrix.npy", normalized_matrix)

    result = {
        "sum": matrix_sum,
        "avr": matrix_average,
        "sumMD": MD_sum,
        "avrMD": MD_avg,
        "sumSD": SD_sum,
        "avrSD": SD_avg,
        "max": max_value,
        "min": min_value
    }

    with open('matrix_stats.json', "w") as file:
        json.dump(result, file, indent=4, cls=NpEncoder)

doing_staff()