from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class UserBase(BaseModel):
    username: str

class UserCreate(UserBase):
    pass

class User(UserBase):
    id: int
    created_at: datetime
    class Config:
        from_attributes = True

class MovieBase(BaseModel):
    title: str
    genres: Optional[str] = None
    poster_url: Optional[str] = None
    description: Optional[str] = None
    year: Optional[int] = None
    presentation_score: Optional[int] = 0

class MovieCreate(MovieBase):
    pass

class Movie(MovieBase):
    id: int
    class Config:
        from_attributes = True

class RatingBase(BaseModel):
    movie_id: int
    rating: Optional[float] = None

class RatingCreate(RatingBase):
    pass

class Rating(RatingBase):
    id: int
    user_id: int
    created_at: datetime
    class Config:
        from_attributes = True

class RecommendationBase(BaseModel):
    movie_id: int
    score: float

class Recommendation(RecommendationBase):
    id: int
    user_id: int
    created_at: datetime
    class Config:
        from_attributes = True
