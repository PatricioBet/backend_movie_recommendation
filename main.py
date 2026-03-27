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

# Create tables
models.Base.metadata.create_all(bind=database.engine)

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
    movies = db.query(models.Movie).order_by(func.random()).limit(limit).all()
    # Mock fallback if db is empty
    if not movies:
        return [{"id": i, "title": f"Hardcoded Movie {i}", "genres": "Action", "year": 2026} for i in range(limit)]
    return movies

@app.post("/users/{user_id}/ratings/", response_model=schemas.Rating)
def create_rating(user_id: int, rating: schemas.RatingCreate, db: Session = Depends(get_db)):
    new_rating = models.Rating(**rating.model_dump(), user_id=user_id)
    existing = db.query(models.Rating).filter(
        models.Rating.user_id == user_id, 
        models.Rating.movie_id == rating.movie_id
    ).first()
    
    if existing:
        existing.rating = rating.rating
        db.commit()
        db.refresh(existing)
        return existing
        
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)
    return new_rating

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
