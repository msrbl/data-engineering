import pandas as pd
import sqlite3
import json

product_file = "_product_data.csv"
update_file = "_update_data.text"

def load_products_to_db(db_name, file_path):
    data = pd.read_csv(file_path, delimiter=";")
    with sqlite3.connect(db_name) as conn:
        data.to_sql("products", conn, if_exists="replace", index=False)
        cursor = conn.cursor()
        cursor.execute("""
            ALTER TABLE products ADD update_count INTEGER DEFAULT 0
            """)
        conn.commit()


def apply_updates(db_name, update_file):
    with sqlite3.connect(db_name) as conn, open(update_file, "r", encoding="utf-8") as file:
        cursor = conn.cursor()
        updates = file.read().split("=====\n")

        for update in updates:
            if not update.strip():
                continue

            update_info = {}
            for line in update.strip().split("\n"):
                key, value = line.split("::")
                update_info[key] = value

            name = update_info["name"]
            method = update_info["method"]
            param = update_info["param"]

            try:
                cursor.execute("BEGIN TRANSACTION")

                if method == "price_abs":
                    cursor.execute("UPDATE products SET price = price + ?, update_count = update_count + 1 WHERE name = ?", (float(param), name))
                elif method == "price_percent":
                    cursor.execute("UPDATE products SET price = price + (price * ?), update_count = update_count + 1 WHERE name = ?", (float(param), name))
                elif method == "quantity_add" or method == "quantity_sub":
                    cursor.execute("UPDATE products SET quantity = quantity + ?, update_count = update_count + 1 WHERE name = ?", (int(param), name))
                elif method == "available":
                    cursor.execute("UPDATE products SET isAvailable = ?, update_count = update_count + 1 WHERE name = ?", (param == "True", name))

                cursor.execute("SELECT price, quantity FROM products WHERE name = ?", (name,))
                row = cursor.fetchone()
                if row and (row[0] < 0 or row[1] < 0):
                    raise ValueError("Price or quantity cannot be negative.")

                cursor.execute("COMMIT")
            except Exception as e:
                cursor.execute("ROLLBACK")
                print(f"Error processing update for {name}: {e}")


def query_top_updated_products(db_name):
    query = "SELECT name, update_count FROM products ORDER BY update_count DESC LIMIT 10"
    with sqlite3.connect(db_name) as conn:
        return conn.execute(query).fetchall()

def analyze_prices(db_name):
    query = """
        SELECT category, COUNT(*) AS count, SUM(price) AS total, MIN(price) AS min_price,
               MAX(price) AS max_price, AVG(price) AS avg_price
        FROM products
        GROUP BY category
    """
    with sqlite3.connect(db_name) as conn:
        return conn.execute(query).fetchall()

def analyze_quantities(db_name):
    query = """
        SELECT category, SUM(quantity) AS total, MIN(quantity) AS min_quantity,
               MAX(quantity) AS max_quantity, AVG(quantity) AS avg_quantity
        FROM products
        GROUP BY category
    """
    with sqlite3.connect(db_name) as conn:
        return conn.execute(query).fetchall()

def query_available_categories(db_name):
    query = "SELECT category, COUNT(*) AS available_count FROM products WHERE isAvailable = 1 GROUP BY category"
    with sqlite3.connect(db_name) as conn:
        return conn.execute(query).fetchall()


def save_results_to_json(results, output_file):
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    database_name = "products.db"

    load_products_to_db(database_name, product_file)
    apply_updates(database_name, update_file)

    top_updated_output_file = "top_updated.json"
    top_updated_rows = query_top_updated_products(database_name)
    save_results_to_json(top_updated_rows, top_updated_output_file)

    analyzed_prices_output_file = "analyzed_prices.json"
    analyzed_prices_rows = analyze_prices(database_name)
    save_results_to_json(analyzed_prices_rows, analyzed_prices_output_file)

    analyzed_quantities_output_file = "analyzed_quantities.json"
    analyzed_quantities_rows = analyze_quantities(database_name)
    save_results_to_json(analyzed_quantities_rows, analyzed_quantities_output_file)

    available_categories_output_file = "available_categories.json"
    available_categories_rows = query_available_categories(database_name)
    save_results_to_json(available_categories_rows, available_categories_output_file)