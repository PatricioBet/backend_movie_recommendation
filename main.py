import os
import pickle
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
import models, schemas, database
from typing import List

# ML additions
import torch
from ncf_model import NCF

from fastapi.middleware.cors import CORSMiddleware

import time
from sqlalchemy.exc import OperationalError

# Create tables with retry for docker-compose startup
max_retries = 5
for i in range(max_retries):
    try:
        models.Base.metadata.create_all(bind=database.engine)
        
        # Automatic mini-migrations
        from sqlalchemy import text
        with database.engine.connect() as conn:
            try:
                conn.execute(text("ALTER TABLE movies ADD COLUMN IF NOT EXISTS presentation_score INTEGER DEFAULT 0"))
                conn.execute(text("ALTER TABLE movies ADD COLUMN IF NOT EXISTS good_rating_count INTEGER DEFAULT 0"))
                conn.execute(text("ALTER TABLE movies ADD COLUMN IF NOT EXISTS bad_rating_count INTEGER DEFAULT 0"))
                conn.commit()
            except Exception as ex:
                print(f"Migration error: {ex}")
            try:
                conn.execute(text("ALTER TABLE ratings ALTER COLUMN rating DROP NOT NULL"))
                conn.commit()
            except Exception as ex:
                print(f"Migration error: {ex}")
                
        print("Database connected and tables created successfully.")
        break
    except OperationalError as e:
        if i == max_retries - 1:
            print("Failed to connect to the database after several retries.")
            raise e
        print(f"Database connection failed. Retrying in 3 seconds... ({i+1}/{max_retries})")
        time.sleep(3)

app = FastAPI(title="Movie Recommendation API")

# Add CORS Middleware to allow requests from Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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
        user2idx = encoders['user2idx']
        movie2idx = encoders['movie2idx']
        
    num_users = len(user2idx)
    num_items = len(movie2idx)
    ncf_net = NCF(num_users=num_users, num_items=num_items, embedding_dim=64, dropout=0.33)
    # Map model to CPU since backend server may not have GPU available
    ncf_net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
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
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
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
    
    # 60% based on presentation_score weighted random
    score_limit = int(limit * 0.3)
    weighted_movies = db.query(models.Movie)\
                        .filter(models.Movie.presentation_score > 0)\
                        .order_by(-func.log(func.random()) / models.Movie.presentation_score)\
                        .limit(score_limit).all()
                        
    actual_score_count = len(weighted_movies)
    random_limit = limit - actual_score_count
    
    # 40% (plus any shortfall) purely random from the rest
    weighted_ids = [m.id for m in weighted_movies]
    query = db.query(models.Movie)
    if weighted_ids:
        query = query.filter(~models.Movie.id.in_(weighted_ids))
        
    random_movies = query.order_by(func.random()).limit(random_limit).all()
    
    movies = weighted_movies + random_movies
    random.shuffle(movies)
    
    # Mock fallback if db is empty
    if not movies:
        return [{"id": i, "title": f"Hardcoded Movie {i}", "genres": "Action", "year": 2026, "presentation_score": 0} for i in range(limit)]
    return movies

@app.post("/users/{user_id}/ratings/", response_model=schemas.Rating)
def create_rating(user_id: int, rating: schemas.RatingCreate, db: Session = Depends(get_db)):
    # Increment presentation score if it's an actual rating (not null/"no vista")
    if rating.rating is not None:
        movie = db.query(models.Movie).filter(models.Movie.id == rating.movie_id).first()
        if movie:
            movie.presentation_score = (movie.presentation_score or 0) + 1
            if rating.rating >= 4.0:
                movie.good_rating_count = (movie.good_rating_count or 0) + 1
            elif rating.rating <= 2.0:
                movie.bad_rating_count = (movie.bad_rating_count or 0) + 1
            db.commit()

    existing = db.query(models.Rating).filter(
        models.Rating.user_id == user_id, 
        models.Rating.movie_id == rating.movie_id
    ).first()
    
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
def rate_recommendation(movie_id: int, rating: schemas.RecRatingCreate, db: Session = Depends(get_db)):
    movie = db.query(models.Movie).filter(models.Movie.id == movie_id).first()
    if movie:
        if rating.is_good:
            movie.good_rating_count = (movie.good_rating_count or 0) + 1
        else:
            movie.bad_rating_count = (movie.bad_rating_count or 0) + 1
        db.commit()
        return {"status": "ok"}
    raise HTTPException(status_code=404, detail="Movie not found")

@app.get("/movies/trending", response_model=List[schemas.Movie])
def get_trending_movies(limit: int = 5, db: Session = Depends(get_db)):
    movies = db.query(models.Movie).filter(
        (models.Movie.good_rating_count > 0) | (models.Movie.bad_rating_count > 0)
    ).all()
    
    def acceptance_score(m):
        good = m.good_rating_count or 0
        bad = m.bad_rating_count or 0
        total = good + bad
        if total == 0:
            return 0
        percentage = good / total
        return (percentage, total)
        
    movies.sort(key=acceptance_score, reverse=True)
    return movies[:limit]

@app.get("/users/{user_id}/recommendations", response_model=List[schemas.Movie])
def get_recommendations(user_id: int, db: Session = Depends(get_db)):
    if ncf_net is not None:
        # If user is new (cold start), map to index 0 dynamically or handle properly
        idx_user = user2idx.get(user_id, 0)
        
        # Consider top 100 random popular movies
        all_movies = db.query(models.Movie).limit(100).all()
        if not all_movies:
            return [{"id": 1, "title": "Inception", "genres": "Sci-Fi", "year": 2010}]
            
        m_ids = [m.id for m in all_movies if m.id in movie2idx]
        if not m_ids:
            return all_movies[:5]
            
        with torch.no_grad():
            u_tensors = torch.tensor([idx_user]*len(m_ids), dtype=torch.long)
            i_tensors = torch.tensor([movie2idx[m] for m in m_ids], dtype=torch.long)
            preds = ncf_net(u_tensors, i_tensors)
            
            top_indices = torch.argsort(preds, descending=True)[:5]
            top_m_ids = [m_ids[i] for i in top_indices.tolist()]
            
        recs = db.query(models.Movie).filter(models.Movie.id.in_(top_m_ids)).all()
        return recs
        
    # Simulation mode fallback
    from sqlalchemy.sql.expression import func
    recs = db.query(models.Movie).order_by(func.random()).limit(5).all()
    if not recs:
        return [{"id": 1, "title": "Simulated The Matrix", "genres": "Sci-Fi", "year": 1999}]
    return recs
