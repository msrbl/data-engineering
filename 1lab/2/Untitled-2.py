def process(input_file, output_file):
    try:
        averages = []

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            numbers = list(map(float, line.strip().split(' ')))

            positive_numbers = [num for num in numbers if num > 0]

            if positive_numbers:
                avg = sum(positive_numbers) / len(positive_numbers)
            else:
                avg = 0

            averages.append(avg)
        
        max_val = max(averages)
        min_val = min(averages)

        with open(output_file, 'w', encoding='utf-8') as file:
            for avg in averages:
                file.write(f"{avg:.2f}\n")
            file.write("\n")
            file.write(f"{max_val:.2f}\n")
            file.write(f"{min_val:.2f}\n")

        print(f"Обработка завершена. Результаты сохранены в '{output_file}'.")
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

input_file = 'second_task.txt'
output_file = 'second_task_output.txt'

process(input_file, output_file)