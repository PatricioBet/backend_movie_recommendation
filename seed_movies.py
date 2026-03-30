import urllib.request
import zipfile
import os
import csv
import re
from database import SessionLocal, engine, DATABASE_URL
import models

def seed():
    models.Base.metadata.create_all(bind=engine)

    print(f"Using database: {DATABASE_URL}")

    print("Downloading MovieLens dataset...")
    url = "https://files.grouplens.org/datasets/movielens/ml-latest.zip"
    zip_path = "ml-latest.zip"
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")

    print("Seeding database...")
    db = SessionLocal()

    before_count = db.query(models.Movie).count()
    inserted_count = 0
    skipped_count = 0

    # Get existing movie IDs to avoid duplicates
    existing_ids = {movie_id for (movie_id,) in db.query(models.Movie.id).all()}
    
    movies_csv_path = "ml-latest/movies.csv"
    if not os.path.exists(movies_csv_path):
        db.close()
        raise FileNotFoundError(f"Could not find dataset file: {movies_csv_path}")

    with open(movies_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        movies_to_insert = []
        for row in reader:
            m_id = int(row['movieId'])
            if m_id in existing_ids:
                skipped_count += 1
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
            inserted_count += 1
            
            if len(movies_to_insert) >= 1000:
                db.bulk_save_objects(movies_to_insert)
                db.commit()
                movies_to_insert = []

        if movies_to_insert:
            db.bulk_save_objects(movies_to_insert)
            db.commit()

    after_count = db.query(models.Movie).count()
    db.close()
    print(
        "Seeding complete! "
        f"Inserted: {inserted_count}, Skipped existing: {skipped_count}, "
        f"Movies before: {before_count}, after: {after_count}."
    )

if __name__ == "__main__":
    seed()
