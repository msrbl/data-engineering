import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(file_path):
    return pd.read_csv(file_path)

def get_file_size(file_path):
    return os.path.getsize(file_path)

def analyze_memory_usage(data):
    memory_usage = data.memory_usage(deep=True).sum()
    column_stats = [
        {
            "column": column,
            "memory_usage": data[column].memory_usage(deep=True),
            "percentage": data[column].memory_usage(deep=True) / memory_usage * 100,
            "dtype": str(data[column].dtype),
        }
        for column in data.columns
    ]
    return sorted(column_stats, key=lambda x: x["memory_usage"], reverse=True), memory_usage

def save_to_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def optimize_data_types(data):
    for column in data.select_dtypes(include=['object']).columns:
        if data[column].nunique() / len(data[column]) < 0.5:
            data[column] = data[column].astype('category')

    for column in data.select_dtypes(include=['int']).columns:
        data[column] = pd.to_numeric(data[column], downcast='integer')

    for column in data.select_dtypes(include=['float']).columns:
        data[column] = pd.to_numeric(data[column], downcast='float')

    return data

def save_filtered_data(file_path, output_file, columns, chunk_size=10000):
    chunk_iter = pd.read_csv(file_path, usecols=columns, chunksize=chunk_size)
    pd.concat(chunk_iter).to_csv(output_file, index=False)

def visualize_data(data, output_file):
    plt.figure(figsize=(15, 10))

    # Line plot
    plt.subplot(2, 3, 1)
    data['income'].plot(kind='line', title='Income')

    # Bar chart
    plt.subplot(2, 3, 2)
    data['fraud_bool'].value_counts().plot(kind='bar', title='Fraud Indicator')

    # Pie chart
    plt.subplot(2, 3, 3)
    data['housing_status'].value_counts().plot(kind='pie', title='Housing Status')

    # Correlation matrix
    plt.subplot(2, 3, 4)
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix')

    # Histogram
    plt.subplot(2, 3, 5)
    data['customer_age'].plot(kind='hist', bins=20, title='Customer Age')

    # Scatter plot
    plt.subplot(2, 3, 6)
    plt.scatter(data['income'], data['credit_risk_score'], alpha=0.5)
    plt.title('Income vs Credit Risk Score')
    plt.xlabel('Income')
    plt.ylabel('Credit Risk Score')

    plt.tight_layout()
    plt.savefig(output_file)

def main():
    input_file = 'bank_dataset.csv'
    column_stats_file = 'column_stats_unoptimized.json'
    memory_comparison_file = 'memory_comparison.json'
    filtered_data_file = 'filtered_data.csv'
    plots_file = 'plots.png'

    data = load_data(input_file)

    column_stats, original_memory = analyze_memory_usage(data)
    save_to_json(column_stats, column_stats_file)

    data = optimize_data_types(data)
    _, optimized_memory = analyze_memory_usage(data)

    memory_comparison = {
        "Original memory usage": int(original_memory),
        "Optimized memory usage": int(optimized_memory)
    }
    save_to_json(memory_comparison, memory_comparison_file)

    selected_columns = [
        "fraud_bool", "income", "name_email_similarity", "customer_age", "payment_type", 
        "zip_count_4w", "employment_status", "credit_risk_score", "housing_status", "source"
    ]
    save_filtered_data(input_file, filtered_data_file, selected_columns)

    visualize_data(data, plots_file)

if __name__ == "__main__":
    main()