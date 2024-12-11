import requests

def fetch_data_and_convert_to_html(api_url, output_html_file):
    try:
        response = requests.get(api_url)
        
        data = response.json()

        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>API Data</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f4f4f4; }
            </style>
        </head>
        <body>
            <h1>API Data Representation from JSON to HTML</h1>
            <table>
                <thead>
                    <tr>
                        <th>User ID</th>
                        <th>Post ID</th>
                        <th>Title</th>
                        <th>Body</th>
                    </tr>
                </thead>
                <tbody>
        """

        for post in data[:200]:
            html_content += f"""
                <tr>
                    <td>{post['userId']}</td>
                    <td>{post['id']}</td>
                    <td>{post['title']}</td>
                    <td>{post['body']}</td>
                </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        with open(output_html_file, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        print(f"Данные успешно сохранены в '{output_html_file}'")

    except Exception as e:
        print(f"Произошла ошибка: {e}")

api_url = "https://jsonplaceholder.typicode.com/posts"
output_html_file = "sixth_task_result.html"

fetch_data_and_convert_to_html(api_url, output_html_file)