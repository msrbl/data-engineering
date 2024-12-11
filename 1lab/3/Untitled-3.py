import math

def process_na_file(input_file, output_file):
    try:
        results = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            items = line.split()

            numbers = []

            for i, item in enumerate(items):
                if item == 'N/A':
                    left = float(numbers[i - 1]) if i > 0 else None
                    right = float(items[i + 1]) if i < len(items) - 1 and items[i + 1] != 'N/A' else None

                    if left is not None and right is not None:
                        numbers.append((left + right) / 2)
                    elif left is not None:
                        numbers.append(left)
                    elif right is not None:
                        numbers.append(right)
                    else:
                        numbers.append(0)
                else:
                    numbers.append(float(item))

            valid_numbers = [num for num in numbers if num > 0 and math.sqrt(num) > 50]

            line_sum = sum(valid_numbers)
            results.append(line_sum)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"{result:.2f}\n")
        
        print(f"Обработка завершена. Результаты сохранены в '{output_file}'.")
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

input_file = 'third_task.txt'
output_file = 'third_task_output.txt'

process_na_file(input_file, output_file)