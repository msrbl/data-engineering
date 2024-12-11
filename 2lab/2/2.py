import numpy as np
import os

matrix = np.load('second_task.npy')

x, y, z = [], [], []
npz_path = "result_arrays.npz"
compressed_npz_path = "compressed_result_arrays.npz"

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if matrix[i, j] > 524:
            x.append(i)
            y.append(j)
            z.append(matrix[i, j])

x = np.array(x)
y = np.array(y)
z = np.array(z)

np.savez(npz_path, x=x, y=y, z=z)
np.savez_compressed(compressed_npz_path, x=x, y=y, z=z)

npz_size = os.path.getsize(npz_path)
compressed_npz_size = os.path.getsize(compressed_npz_path)
diff = npz_size / compressed_npz_size

result = f'''
Размер не сжатого файла: {npz_size} байт
Размер сжатого файла: {compressed_npz_size} байт

Сжатие уменьшило вес в {diff:.2f} раз
'''


with open("comparison_sizes.txt", "w", encoding="utf-8") as f:
    f = f.write(result)