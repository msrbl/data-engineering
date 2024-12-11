from collections import Counter
import re

def count_words(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        words = re.findall(r'\b\w+\b', text.lower())

        word_counts = Counter(words)

        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as file:
            for word, freq in sorted_word_counts:
                file.write(f"{word}:{freq}\n")
        
        print(f"Обработка завершена. Результаты сохранены в '{output_file}'.")
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

def count_words_over4(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        words = re.findall(r'\b\w+\b', text.lower())

        filtered_words = [word for word in words if len(word) > 4]

        word_counts = Counter(filtered_words)

        words_proportion = float(len(filtered_words) / len(words))
        words_sum = len(filtered_words)

        sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(f"count:{words_sum}\n")
            file.write(f"proportion:{words_proportion:.2f}\n\n")
            for word, freq in sorted_word_counts:
                file.write(f"{word}:{freq}\n")
        
        print(f"Обработка завершена. Результаты сохранены в '{output_file}'.")
    
    except FileNotFoundError:
        print(f"Ошибка: Файл '{input_file}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

input_file = 'first_task.txt'
output_file1 = 'first_task_output.txt'
output_file2 = 'first_task_output_4.txt'

count_words(input_file, output_file1)
count_words_over4(input_file, output_file2)