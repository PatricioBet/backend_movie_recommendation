import os
import pickle
from typing import List

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import text, func
from sqlalchemy.orm import Session

import database
import models
import schemas

# ML additions
import torch
from ncf_model import NCF

from fastapi.middleware.cors import CORSMiddleware

import time
from sqlalchemy.exc import OperationalError


def calculate_trending_score(good_count: int, bad_count: int) -> float:
    # Laplace smoothing to reduce volatility with very low vote counts.
    return (good_count + 1) / (good_count + bad_count + 2)


def update_movie_trending_score(movie: models.Movie) -> None:
    good = movie.good_rating_count or 0
    bad = movie.bad_rating_count or 0
    movie.trending_score = calculate_trending_score(good, bad)


# Create tables with retry for container startup.
max_retries = int(os.getenv("DB_MAX_RETRIES", "20"))
retry_delay_seconds = int(os.getenv("DB_RETRY_DELAY_SECONDS", "3"))
for i in range(max_retries):
    try:
        models.Base.metadata.create_all(bind=database.engine)
        break  # conexión OK, salir del loop
    except OperationalError as e:
        if i == max_retries - 1:
            print("Failed to connect to the database after several retries.")
            raise e
        print(
            f"Database connection failed. Retrying in {retry_delay_seconds} seconds... ({i + 1}/{max_retries})"
        )
        time.sleep(retry_delay_seconds)

# Migraciones — solo una vez, fuera del loop
with database.engine.connect() as conn:
    try:
        conn.execute(
            text(
                "ALTER TABLE movies ADD COLUMN IF NOT EXISTS presentation_score INTEGER DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE movies ADD COLUMN IF NOT EXISTS good_rating_count INTEGER DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE movies ADD COLUMN IF NOT EXISTS bad_rating_count INTEGER DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "ALTER TABLE movies ADD COLUMN IF NOT EXISTS trending_score FLOAT DEFAULT 0"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS idx_movies_trending_score ON movies (trending_score DESC)"
            )
        )
        conn.commit()
    except Exception as ex:
        print(f"Migration warning: {ex}")
    try:
        conn.execute(text("ALTER TABLE ratings ALTER COLUMN rating DROP NOT NULL"))
        conn.commit()
    except Exception as ex:
        print(f"Migration warning: {ex}")

print("Database connected and tables created successfully.")

app = FastAPI(title="Movie Recommendation API")

# Add CORS Middleware to allow requests from Next.js
cors_origins_raw = os.getenv("CORS_ORIGINS", "*")
cors_origins = [
    origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()
]
allow_all_origins = len(cors_origins) == 1 and cors_origins[0] == "*"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=not allow_all_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML Model
MODEL_PATH = "/app/models/ncf_model.pth"
ENCODERS_PATH = "/app/models/encoders.pkl"
ml_ready = False
ncf_net = None
user2idx = {}
movie2idx = {}

try:
    with open(ENCODERS_PATH, "rb") as f:
        encoders = pickle.load(f)
        user2idx = encoders["user2idx"]
        movie2idx = encoders["movie2idx"]

    num_users = len(user2idx)
    num_items = len(movie2idx)
    # Keep architecture aligned with the trained checkpoint to avoid shape mismatches.
    ncf_net = NCF(
        num_users=num_users, num_items=num_items, embedding_dim=64, dropout=0.33
    )
    # Map model to CPU since backend server may not have GPU available
    ncf_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    ncf_net.eval()
    ml_ready = True
    print(f"ML Model loaded successfully. Users: {num_users}, Items: {num_items}")
except Exception as e:
    print("ML Model not loaded. Backend will run in simulation mode. Error:", e)


def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def read_root():
    return {"message": "Welcome to the Movie Recommendation API"}


@app.post("/users/", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = (
        db.query(models.User).filter(models.User.username == user.username).first()
    )
    if db_user:
        return db_user
    new_user = models.User(username=user.username)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.get("/movies/random", response_model=List[schemas.Movie])
def get_random_movies(limit: int = 10, db: Session = Depends(get_db)):
    from sqlalchemy.sql.expression import func
    import random

    score_limit = int(limit * 0.1)
    weighted_movies = (
        db.query(models.Movie)
        .filter(models.Movie.presentation_score > 0)
        .order_by(-func.log(func.random()) / models.Movie.presentation_score)
        .limit(score_limit)
        .all()
    )

    actual_score_count = len(weighted_movies)
    random_limit = limit - actual_score_count

    weighted_ids = [m.id for m in weighted_movies]
    query = db.query(models.Movie)
    if weighted_ids:
        query = query.filter(~models.Movie.id.in_(weighted_ids))

    random_movies = query.order_by(func.random()).limit(random_limit).all()

    movies = weighted_movies + random_movies
    random.shuffle(movies)

    # Mock fallback if db is empty
    if not movies:
        return [
            {
                "id": i,
                "title": f"Película {i}",
                "genres": "Action",
                "year": 2026,
                "presentation_score": 0,
            }
            for i in range(limit)
        ]
    return movies


@app.post("/users/{user_id}/ratings/", response_model=schemas.Rating)
def create_rating(
    user_id: int, rating: schemas.RatingCreate, db: Session = Depends(get_db)
):
    movie = db.query(models.Movie).filter(models.Movie.id == rating.movie_id).first()
    if not movie:
        raise HTTPException(status_code=404, detail="Movie not found")

    existing = (
        db.query(models.Rating)
        .filter(
            models.Rating.user_id == user_id, models.Rating.movie_id == rating.movie_id
        )
        .first()
    )

    old_rating = existing.rating if existing else None

    if old_rating is not None:
        if old_rating >= 4.0:
            movie.good_rating_count = max((movie.good_rating_count or 0) - 1, 0)
        elif old_rating <= 2.0:
            movie.bad_rating_count = max((movie.bad_rating_count or 0) - 1, 0)

    if rating.rating is not None:
        movie.presentation_score = (movie.presentation_score or 0) + 1
        if rating.rating >= 4.0:
            movie.good_rating_count = (movie.good_rating_count or 0) + 1
        elif rating.rating <= 2.0:
            movie.bad_rating_count = (movie.bad_rating_count or 0) + 1

    update_movie_trending_score(movie)

    if existing:
        existing.rating = rating.rating
        db.commit()
        db.refresh(existing)
        return existing

    new_rating = models.Rating(**rating.model_dump(), user_id=user_id)
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    return new_rating


@app.post("/movies/{movie_id}/recommendation-rating/")
def rate_recommendation(
    movie_id: int, rating: schemas.RecRatingCreate, db: Session = Depends(get_db)
):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if movie:
        if rating.is_good:
            movie.good_rating_count = (movie.good_rating_count or 0) + 1
        else:
            movie.bad_rating_count = (movie.bad_rating_count or 0) + 1
        update_movie_trending_score(movie)
        db.commit()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Movie not found")


@app.get("/movies/trending", response_model=List[schemas.Movie])
def get_trending_movies(limit: int = 6, db: Session = Depends(get_db)):
    return (
        db.query(models.Movie)
        .filter(
            (models.Movie.good_rating_count > 0) | (models.Movie.bad_rating_count > 0)
        )
        .order_by(
            models.Movie.trending_score.desc(),
            models.Movie.good_rating_count.desc(),
            models.Movie.id.asc(),
        )
        .limit(limit)
        .all()
    )


@app.get("/users/{user_id}/recommendations", response_model=List[schemas.Movie])
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    if ncf_net is None:
        recs = db.query(models.Movie).order_by(func.random()).limit(5).all()
        return recs or [
            {"id": 1, "title": "Simulated The Matrix", "genres": "Sci-Fi", "year": 1999}
        ]

    idx_user = user2idx.get(user_id, 0)

    # ✅ Solo traer id + conteos, solo películas conocidas por el modelo
    rows = (
        db.query(
            models.Movie.id,
            models.Movie.good_rating_count,
            models.Movie.bad_rating_count,
        )
        .filter(models.Movie.id.in_(list(movie2idx.keys())))
        .all()
    )

    if not rows:
        return db.query(models.Movie).limit(5).all()

    m_ids = [r.id for r in rows]
    good_arr = [r.good_rating_count or 0 for r in rows]
    bad_arr = [r.bad_rating_count or 0 for r in rows]

    with torch.no_grad():
        u_t = torch.tensor([idx_user] * len(m_ids), dtype=torch.long)
        i_t = torch.tensor([movie2idx[m] for m in m_ids], dtype=torch.long)

        preds = ncf_net(u_t, i_t)  # shape: (N,)

        # ✅ Vectorizado: calcula trending score sin loop
        good_t = torch.tensor(good_arr, dtype=torch.float32)
        bad_t = torch.tensor(bad_arr, dtype=torch.float32)
        total = good_t + bad_t + 1e-8
        wilson_t = (good_t / total) * torch.log1p(
            total
        )  # tu trending score vectorizado
        wilson_norm = wilson_t / (wilson_t.max() + 1e-8)

        adjusted = preds * 0.7 + wilson_norm * 0.3 * preds

        # ✅ torch.topk evita ordenar todo el array
        top_k = torch.topk(adjusted, k=min(5, len(m_ids)))
        top_idxs = top_k.indices.tolist()

    top_m_ids = [m_ids[i] for i in top_idxs]

    recs = db.query(models.Movie).filter(models.Movie.id.in_(top_m_ids)).all()
    recs_dict = {m.id: m for m in recs}
    return [recs_dict[m_id] for m_id in top_m_ids if m_id in recs_dict]
