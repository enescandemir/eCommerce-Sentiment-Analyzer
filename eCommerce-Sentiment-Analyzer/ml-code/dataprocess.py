import sys
import base64
import csv
import string
import re
import pyodbc

stopwords = [
    "ve", "veya", "ile", "fakat", "ancak", "çünkü", "ki", "de", "da", "gibi", "için", "ya da", "yalnız", "ama",
    "ben", "sen", "o", "biz", "siz", "onlar", "bu", "şu", "o", "hepsi", "herkes", "hiçbiri", "bazıları", "çok",
    "az", "daha", "en", "oldukça", "kadar", "tüm", "her", "hiç", "bir", "başka", "pek", "neredeyse", "çoğu",
    "olmak", "yapmak", "etmek", "demek", "denmek", "bulunmak", "verilmek", "şimdi", "sonra", "önce", "şu an",
    "zaten", "hemen", "halen", "mı", "mi", "mu", "mü", "ise", "şunlar", "hiçbir", "herhangi", "hep", "tüm",
    "yine", "aynı"
]

def normalize_text(text):
    translation_table = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
    return text.translate(translation_table)

normalized_stopwords = [normalize_text(w) for w in stopwords]

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F700-\U0001F77F"  
        u"\U0001F780-\U0001F7FF"  
        u"\U0001F800-\U0001F8FF"  
        u"\U0001F900-\U0001F9FF"  
        u"\U0001FA00-\U0001FA6F"  
        u"\U00002702-\U000027B0"  
        u"\U000024C2-\U0001F251" 
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    words = text.split()
    return ' '.join(word for word in words if
                    word.lower() not in stopwords and normalize_text(word.lower()) not in normalized_stopwords)

def reduce_character_repetition(text):
    return re.sub(r'(.)\1{2,}', r'\1', text)


def remove_punctuation_with_space(text):
    return re.sub(r'([.!?,])(\S)', r'\1 \2', text)


def process_file(file_path, lowercase=False, remove_numbers=False, remove_punctuation=False,
                 remove_emojis_flag=False, char_repetition_reduction=False, stopword_removal=False):
    output_lines = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            processed_row = []
            for i, field in enumerate(row):
                if i == 2:
                    field = normalize_text(field)
                    if lowercase:
                        field = field.lower()
                    if remove_numbers:
                        field = re.sub(r'\d+', '', field)
                    if remove_punctuation:
                        field = remove_punctuation_with_space(field)
                        field = field.translate(str.maketrans('', '', string.punctuation))
                    if remove_emojis_flag:
                        field = remove_emojis(field)
                    if char_repetition_reduction:
                        field = reduce_character_repetition(field)
                    if stopword_removal:
                        field = remove_stopwords(field)
                processed_row.append(field)
            output_lines.append(processed_row)
    return output_lines

def save_to_db(processed_data):
    connection_string = "Driver={ODBC Driver 17 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;"
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM ProcessedComments")
    conn.commit()

    cursor.execute("DBCC CHECKIDENT ('ProcessedComments', RESEED, 0)")
    conn.commit()

    for row in processed_data:
        if len(row) < 3:
            print(f"Satırda yeterli sütun yok: {row}")
            continue

        try:
            product_id = int(row[1]) 
            comment_context = row[2]  
        except ValueError:
            print(f"Geçersiz sayı formatı: {row}")
            continue

        cursor.execute("""
            INSERT INTO ProcessedComments (Product_ID, Comment_Context, LabeledBy)
            VALUES (?, ?, NULL)
        """, (product_id, comment_context))
        conn.commit()

    conn.close()

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')

    file_content = sys.argv[1]
    lowercase = sys.argv[2].lower() == 'true'
    remove_numbers = sys.argv[3].lower() == 'true'
    remove_punctuation = sys.argv[4].lower() == 'true'
    remove_emojis_flag = sys.argv[5].lower() == 'true'
    stopword_removal = sys.argv[6].lower() == 'true'
    char_repetition_reduction = sys.argv[7].lower() == 'true'

    processed_data = process_file(file_content, lowercase, remove_numbers, remove_punctuation,
                                  remove_emojis_flag, stopword_removal, char_repetition_reduction)

    save_to_db(processed_data)

    print("Veriler başarıyla işlendi ve veritabanına kaydedildi.")
