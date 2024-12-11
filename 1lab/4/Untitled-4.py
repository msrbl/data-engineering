import csv

def process_csv(input_file, stats_file, modified_file):
    rows = []
    filtered_rows = []
    prices = []

    with open(input_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = [field for field in reader.fieldnames if field != 'expiration_date']
            
        min_quantity = 100000.0
        max_rating = 0

        for row in reader:
            price = float(row['price'])
            prices.append(price)

            min_quantity = min(min_quantity, float(row['quantity']))
            max_rating = max(max_rating, float(row['rating']))

            if float(row['price']) > 665:
                filtered_row = {key: value for key, value in row.items() if key != 'expiration_date'}
                filtered_rows.append(filtered_row)
 
            row.pop('expiration_date', None)
            rows.append(row)

    avg_price = sum(prices) / len(prices)

    with open(stats_file, 'w', encoding='utf-8') as statsfile:
        statsfile.write(f"avg_price: {avg_price:.2f}\n")
        statsfile.write(f"max_rating: {max_rating:.2f}\n")
        statsfile.write(f"min_quantity: {min_quantity:.2f}\n")

    with open(modified_file, 'w', encoding='utf-8', newline='') as modfile:
        writer = csv.DictWriter(modfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(filtered_rows)

input_file = 'fourth_task.txt'
stats_file = 'fourth_task_output_stats.txt'
modified_file = 'fourth_task_output.csv'

process_csv(input_file, stats_file, modified_file)