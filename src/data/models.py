from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///dutch_vocab.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    telegram_id = Column(Integer, unique=True, index=True)
    username = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    learning_sessions = relationship("LearningSession", back_populates="user")

class UserWord(Base):
    __tablename__ = "user_words"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    
    # Word data (previously in Word table)
    dutch_word = Column(String, nullable=False, index=True)
    english_translation = Column(String, nullable=False)
    word_length = Column(Integer)
    
    # Vocabulary management (previously in UserVocabulary)
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Progress tracking (previously in WordProgress)
    mastery_level = Column(Float, default=0.0)  # 0-1 scale, ML predicted
    last_seen = Column(DateTime, nullable=True)
    times_seen = Column(Integer, default=0)
    times_correct = Column(Integer, default=0)
    average_response_time = Column(Float, default=0.0)
    preferred_exercise_type = Column(String, nullable=True)  # bandits recommendation
    
    # Spaced repetition data (from LearningSession)
    next_review_date = Column(DateTime, nullable=True)
    repetition_count = Column(Integer, default=0)
    ease_factor = Column(Float, default=2.5)
    
    user = relationship("User", backref="words")

class LearningSession(Base):
    __tablename__ = "learning_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    user_word_id = Column(Integer, ForeignKey("user_words.id"))  # Reference to UserWord instead of Word
    exercise_type = Column(String)  # 'multiple_choice', 'translation_en_to_nl', 'translation_nl_to_en'
    is_correct = Column(Boolean)
    response_time = Column(Float)  # in seconds
    confidence_score = Column(Float, default=0.5)  # 0-1 scale
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="learning_sessions")
    user_word = relationship("UserWord", backref="learning_sessions")

class BanditModel(Base):
    __tablename__ = "bandit_models"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    exercise_type = Column(String)  # 'multiple_choice_en_to_nl', etc.
    model_coefficients = Column(Text)  # JSON string of model coefficients
    model_intercept = Column(Float)
    scaler_mean = Column(Text)  # JSON string of scaler means
    scaler_scale = Column(Text)  # JSON string of scaler scales
    is_trained = Column(Boolean, default=False)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", backref="bandit_models")

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()