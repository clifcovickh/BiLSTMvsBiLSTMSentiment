import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_google_news_monthly(keyword, start_date, end_date):
    print(f"Mencari '{keyword}' periode {start_date} s/d {end_date}...")
    
    # Menggunakan parameter after: dan before: untuk memecah batasan 100 artikel Google
    url = f"https://news.google.com/rss/search?q={keyword}+after:{start_date}+before:{end_date}&hl=id&gl=ID&ceid=ID:id"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, features='xml')
        items = soup.find_all('item') # Warning teratasi dengan find_all
        
        news_data = []
        for item in items:
            title = item.title.text if item.title else ""
            link = item.link.text if item.link else ""
            pub_date = item.pubDate.text if item.pubDate else ""
            
            source = ""
            clean_title = title
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                clean_title = parts[0]
                source = parts[1]
            
            news_data.append({
                'date': pub_date,
                'title': clean_title,
                'source': source,
                'link': link,
                'keyword': keyword
            })
            
        return news_data
        
    except Exception as e:
        print(f"Terjadi error: {e}")
        return []

# Daftar bulan
months_2025 = [
    ("2025-01-01", "2025-01-31"),
    ("2025-02-01", "2025-02-28"),
    ("2025-03-01", "2025-03-31"),
    ("2025-04-01", "2025-04-30"),
    ("2025-05-01", "2025-05-31"),
    ("2025-06-01", "2025-06-30"),
    ("2025-07-01", "2025-07-31"),
    ("2025-08-01", "2025-08-31"),
    ("2025-09-01", "2025-09-30"),
    ("2025-10-01", "2025-10-31"),
    ("2025-11-01", "2025-11-30"),
    ("2025-12-01", "2025-12-31"),
    ("2026-01-01", "2026-01-31"),
    ("2026-02-01", "2026-02-28"),
    ("2026-03-01", "2026-03-06"),

]

keywords = ["Saham BBCA", "Bank BCA", "IHSG BBCA", "Saham BCA"]
all_news = []

# Proses Scraping
for kw in keywords:
    print(f"\n=== Memulai pencarian massal untuk keyword: {kw} ===")
    for start, end in months_2025:
        data = scrape_google_news_monthly(kw, start, end)
        all_news.extend(data)
        time.sleep(1) # Jeda 1 detik agar aman dari blokir Google

# Menggabungkan dan membersihkan hasil akhir
if all_news:
    df = pd.DataFrame(all_news)
    
    # Menghapus duplikat judul jika ada berita yang sama muncul di bulan berdekatan
    df = df.drop_duplicates(subset=['title'])
    
    filename = "GoogleNews_BBCA.csv"
    df.to_csv(filename, index=False)
    print(f"\nSUCCESS BESAR! Tersimpan total {len(df)} berita unik di {filename}")
else:
    print("\nGagal mengambil data.")