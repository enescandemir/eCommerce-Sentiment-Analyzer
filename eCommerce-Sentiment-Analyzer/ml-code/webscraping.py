import sys
import pyodbc
from selenium import webdriver
from time import sleep
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import codecs
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)

connection_string = "Driver={ODBC Driver 17 for SQL Server};Server=YOUR_SERVER;Database=YOUR_DB;Trusted_Connection=yes;"
conn = pyodbc.connect(connection_string)
cursor = conn.cursor()

path = "PATH_TO_CHROMEDRIVER_EXE"
service = Service(executable_path=path)
browser = webdriver.Chrome(service=service)

if len(sys.argv) > 1:
    category_link = sys.argv[1]
else:
    sys.exit()

parsed_url = urlparse(category_link)
query_params = parse_qs(parsed_url.query)

if 'pi' in query_params:
    query_params['pi'] = ['1']

new_query_string = urlencode(query_params, doseq=True)
new_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, new_query_string, parsed_url.fragment))

category_link = new_url if new_query_string else category_link

total_comment_count = 0
max_comment_limit = 10
skipped_products = 0
page_number = 1

while total_comment_count < max_comment_limit:
    paginated_link = f"{category_link}&pi={page_number}"
    browser.get(paginated_link)
    sleep(3)

    product_links = []
    product_elements = browser.find_elements(By.CLASS_NAME, 'p-card-wrppr.with-campaign-view')
    for product_element in product_elements:
        product_link = product_element.find_element(By.TAG_NAME, 'a').get_attribute('href')
        product_links.append(product_link)

    if not product_links:
        break

    for link in product_links:
        cursor.execute("SELECT Product_ID FROM Products WHERE Product_Link = ?", (link,))
        result = cursor.fetchone()

        if result:
            skipped_products += 1
            continue

        if total_comment_count >= max_comment_limit:
            break

        browser.get(link)
        sleep(3)

        try:
            cookie_button = browser.find_element(By.ID, 'onetrust-accept-btn-handler')
            cookie_button.click()
        except:
            pass
        sleep(3)

        try:
            try:
                brand_element = browser.find_element(By.XPATH, '//h1[@class="pr-new-br"]/a')
                brand = brand_element.text
            except:
                brand_element = browser.find_element(By.XPATH, '//h1[@class="pr-new-br"]/span[1]')
                brand = brand_element.text

            name_element = browser.find_element(By.XPATH, '//h1[@class="pr-new-br"]/span[last()]')
            name = name_element.text

            price_element = browser.find_element(By.XPATH, "//div[@class='product-price-container']//span[@class='prc-dsc']")
            price = price_element.text

            try:
                rating_element = browser.find_element(By.CLASS_NAME, 'product-rating-score')
                rating = rating_element.find_element(By.CLASS_NAME, 'value').text
            except:
                skipped_products += 1
                continue

            image_element = browser.find_element(By.XPATH, '//div[@style="position: relative;"]/img')
            image_url = image_element.get_attribute('src')

            cursor.execute("""
                INSERT INTO Products (Product_Name, Product_Brand, Product_Link, Product_Price, Product_Star_Rating, Product_Image_Url) 
                OUTPUT INSERTED.Product_ID
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, brand, link, price, rating, image_url))

            product_id = cursor.fetchone()[0]
            conn.commit()

            try:
                sleep(2)
                search_button = browser.find_element(By.CLASS_NAME, 'navigate-all-reviews-btn')
                browser.execute_script("arguments[0].click();", search_button)

                sleep(3)
                scroll_pause_time = 2

                comment_count = 0
                no_change_counter = 0
                no_change_limit = 3

                while no_change_counter < no_change_limit and total_comment_count < max_comment_limit:
                    comments = browser.find_elements(By.CLASS_NAME, 'comment-text')

                    if len(comments) > comment_count:
                        for comment in comments[comment_count:]:
                            cursor.execute("""
                                INSERT INTO Comments (Product_ID, Comment_Context) 
                                VALUES (?, ?)
                            """, (product_id, comment.text))
                            conn.commit()

                            total_comment_count += 1
                            if total_comment_count >= max_comment_limit:
                                break

                        comment_count = len(comments)
                        no_change_counter = 0
                    else:
                        no_change_counter += 1

                    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                    sleep(scroll_pause_time)
                    browser.execute_script("window.scrollBy(0, -1000);")
                    sleep(scroll_pause_time)

            except:
                pass

        except:
            skipped_products += 1
            pass

    page_number += 1

    if not product_links or total_comment_count >= max_comment_limit:
        break

browser.quit()
conn.close()

print(f"Yorum cekme islemi tamamlandi. Toplamda {total_comment_count} adet yorum basariyla cekildi.")
print(f"Degerlendirilmeyen {skipped_products} urun atlandi.")
