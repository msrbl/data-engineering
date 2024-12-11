from bs4 import BeautifulSoup
import csv

def html_to_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as html_file:
        soup = BeautifulSoup(html_file, 'html.parser')

    table = soup.find('table')

    headers = [th.text.strip() for th in table.find_all('th')]

    rows = []
    for tr in table.find_all('tr'):
        cells = [td.text.strip() for td in tr.find_all('td')]
        if cells:
            rows.append(cells)

    with open(output_file, 'w', encoding='utf-8', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        writer.writerows(rows)

input_file = 'fifth_task.html'
output_file = 'fifth_task_output.csv'

html_to_csv(input_file, output_file)