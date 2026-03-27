import urllib.request
import zipfile
import os
import csv
import re
from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

def seed():
    models.Base.metadata.create_all(bind=engine)

    print("Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    zip_path = "ml-latest-small.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    print("Seeding database...")
    db = SessionLocal()

    # Get existing movie IDs to avoid duplicates
    existing_ids = {m.id for m in db.query(models.Movie.id).all()}
    
    with open("ml-latest-small/movies.csv", newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        movies_to_insert = []
        for row in reader:
            m_id = int(row['movieId'])
            if m_id in existing_ids:
                continue
                
            title_raw = row['title']
            genres = row['genres'].replace('|', ', ')
            year = None
            
            match = re.search(r'\((\d{4})\)', title_raw)
            if match:
                year = int(match.group(1))
                title = title_raw.replace(f"({year})", "").strip()
            else:
                title = title_raw.strip()
                
            movie = models.Movie(
                id=m_id,
                title=title,
                genres=genres,
                year=year,
                poster_url=None, # Leave None to see the fallback emoji, or you could integrate a TMDB API key here
                description="MovieLens database item."
            )
            movies_to_insert.append(movie)
            
            if len(movies_to_insert) >= 1000:
                db.bulk_save_objects(movies_to_insert)
                db.commit()
                movies_to_insert = []

        if movies_to_insert:
            db.bulk_save_objects(movies_to_insert)
            db.commit()

    db.close()
    print("Seeding complete! Real movies are now in the generic database.")

if __name__ == "__main__":
    seed()
