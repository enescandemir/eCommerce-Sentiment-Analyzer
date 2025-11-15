import pyodbc
import csv
import io
import sys

connection_string = "Driver={ODBC Driver 17 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;"
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()


cursor.execute("SELECT Comment_ID, Product_ID, Comment_Context FROM Comments")
comments = cursor.fetchall()

output = io.StringIO()
writer = csv.writer(output)


writer.writerow(['Comment_ID', 'Product_ID', 'Comment_Context'])

for comment in comments:
    writer.writerow([
        comment.Comment_ID,
        comment.Product_ID,
        comment.Comment_Context if comment.Comment_Context is not None else ''
    ])


csv_data = output.getvalue()
output.close()

sys.stdout.reconfigure(encoding='utf-8')
bom = '\ufeff'
print(bom + csv_data)

conn.close()
