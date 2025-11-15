import pyodbc
import csv

connection_string = "Driver={ODBC Driver 17 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;"
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

csv_file_path = 'comments.csv'

with open(csv_file_path, encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)

    for row in reader:
        cursor.execute("""
            INSERT INTO Comments (Product_ID, Comment_Context)
            VALUES (?, ?)
        """, row[1], row[2])

conn.commit()
conn.close()

print("Veriler başarıyla Comments tablosuna yüklendi.")
