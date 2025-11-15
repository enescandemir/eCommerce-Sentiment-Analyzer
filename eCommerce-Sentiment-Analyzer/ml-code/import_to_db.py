import pyodbc
import csv

connection_string = "Driver={ODBC Driver 17 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;"
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

with open('processed_comments.csv', mode='r', encoding='utf-8-sig') as file:
    reader = csv.DictReader(file)

    for row in reader:
        product_id = row['Product_ID']
        comment_context = row['Comment_Context']
        comment_context_ticket = row['Comment_Context_Ticket']

        cursor.execute("""
            INSERT INTO ProcessedComments (Product_ID, Comment_Context, Comment_Context_Ticket)
            VALUES (?, ?, ?)
        """, product_id, comment_context, comment_context_ticket)

conn.commit()

conn.close()

print("Veriler başarıyla veritabanına aktarıldı.")
