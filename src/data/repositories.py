"""
Data access layer for ML models.

This module provides repositories to decouple ML models from direct database access,
making the code more testable and maintainable.
"""

from typing import Any, Dict, List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from data.models import BanditModel, LearningSession, User, UserWord


class UserWordRepository:
    """Repository for UserWord data access"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_word_id: int) -> Optional[UserWord]:
        """Get a user word by ID"""
        return self.db.query(UserWord).filter(UserWord.id == user_word_id).first()

    def get_user_words_with_sessions(self, user_id: int) -> List[UserWord]:
        """Get all user words that have been seen at least once"""
        return self.db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.times_seen > 0
            )
        ).all()

    def get_all_user_words(self, user_id: int) -> List[UserWord]:
        """Get all user words for a user"""
        return self.db.query(UserWord).filter(UserWord.user_id == user_id).all()

    def update_mastery_level(self, user_word_id: int, mastery_level: float):
        """Update the mastery level for a user word"""
        user_word = self.get_by_id(user_word_id)
        if user_word:
            user_word.mastery_level = mastery_level

    def update_average_response_time(self, user_word_id: int, response_time: float, alpha: float = 0.3):
        """Update average response time using exponential moving average"""
        user_word = self.get_by_id(user_word_id)
        if user_word:
            if user_word.average_response_time.is_(0):
                user_word.average_response_time = response_time
            else:
                user_word.average_response_time = (
                    alpha * response_time +
                    (1 - alpha) * user_word.average_response_time
                )


class LearningSessionRepository:
    """Repository for LearningSession data access"""

    def __init__(self, db: Session):
        self.db = db

    def get_word_sessions(self, user_id: int, user_word_id: int) -> List[LearningSession]:
        """Get all sessions for a specific user and word"""
        return self.db.query(LearningSession).filter(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.user_word_id == user_word_id
            )
        ).order_by(LearningSession.timestamp).all()

    def get_user_sessions(self, user_id: int) -> List[LearningSession]:
        """Get all sessions for a user"""
        return self.db.query(LearningSession).filter(
            LearningSession.user_id == user_id
        ).all()

    def get_exercise_type_sessions(self, user_id: int, exercise_type: str) -> List[LearningSession]:
        """Get all sessions for a specific user and exercise type"""
        return self.db.query(LearningSession).filter(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.exercise_type == exercise_type
            )
        ).all()


class BanditModelRepository:
    """Repository for BanditModel data access"""

    def __init__(self, db: Session):
        self.db = db

    def get_model(self, user_id: int, exercise_type: str) -> Optional[BanditModel]:
        """Get a bandit model for a specific user and exercise type"""
        return self.db.query(BanditModel).filter(
            and_(
                BanditModel.user_id == user_id,
                BanditModel.exercise_type == exercise_type
            )
        ).first()

    def save_model(self, user_id: int, exercise_type: str, model_data: Dict[str, Any]) -> BanditModel:
        """Save or update a bandit model"""
        bandit_model = self.get_model(user_id, exercise_type)

        if not bandit_model:
            bandit_model = BanditModel(
                user_id=user_id,
                exercise_type=exercise_type
            )
            self.db.add(bandit_model)

        # Update model parameters
        bandit_model.model_coefficients = model_data['coefficients']
        bandit_model.model_intercept = model_data['intercept']
        bandit_model.scaler_mean = model_data['scaler_mean']
        bandit_model.scaler_scale = model_data['scaler_scale']
        bandit_model.is_trained = model_data['is_trained']
        bandit_model.updated_at = model_data['updated_at']

        self.db.commit()
        return bandit_model

    def load_model_data(self, user_id: int, exercise_type: str) -> Optional[Dict[str, Any]]:
        """Load model data as a dictionary"""
        bandit_model = self.get_model(user_id, exercise_type)

        _a: int = len(exercise_type)

        if not bandit_model or not bandit_model.is_trained:
            return None
        try:
            import json
            return {
                'coefficients': json.loads(bandit_model.model_coefficients),
                'intercept': bandit_model.model_intercept,
                'scaler_mean': json.loads(bandit_model.scaler_mean),
                'scaler_scale': json.loads(bandit_model.scaler_scale),
                'is_trained': bandit_model.is_trained
            }
        except Exception as e:
            print(f"Error loading model from DB: {e}")
            return None


class UserRepository:
    """Repository for User data access"""

    def __init__(self, db: Session):
        self.db = db

    def get_by_telegram_id(self, telegram_id: int) -> Optional[User]:
        """Get user by Telegram ID"""
        return self.db.query(User).filter(User.telegram_id == telegram_id).first()

    def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by internal ID"""
        return self.db.query(User).filter(User.id == user_id).first()


class MLDataService:
    """Service that combines repositories for ML operations"""

    def __init__(self, db: Session):
        self.db = db
        self.user_word_repo = UserWordRepository(db)
        self.session_repo = LearningSessionRepository(db)
        self.bandit_repo = BanditModelRepository(db)
        self.user_repo = UserRepository(db)

    def get_word_training_data(self, user_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get training data for progress prediction model"""
        if user_id:
            user_words = self.user_word_repo.get_user_words_with_sessions(user_id)
        else:
            # Get all user words with sessions (for global training)
            user_words = self.db.query(UserWord).filter(UserWord.times_seen > 0).all()

        training_data = []
        for user_word in user_words:
            word_sessions = self.session_repo.get_word_sessions(user_word.user_id, user_word.id)
            user_sessions = self.session_repo.get_user_sessions(user_word.user_id)

            training_data.append({
                'user_word': user_word,
                'word_sessions': word_sessions,
                'user_sessions': user_sessions,
                'target': 1 if user_word.mastery_level >= 0.7 else 0
            })

        return training_data

    def get_word_prediction_data(self, user_id: int, user_word_id: int) -> Optional[Dict[str, Any]]:
        """Get data needed for making a prediction on a single word"""
        user_word = self.user_word_repo.get_by_id(user_word_id)
        if not user_word:
            return None

        word_sessions = self.session_repo.get_word_sessions(user_id, user_word_id)
        user_sessions = self.session_repo.get_user_sessions(user_id)

        return {
            'user_word': user_word,
            'word_sessions': word_sessions,
            'user_sessions': user_sessions
        }

    def apply_mastery_predictions(self, user_id: int, predictions: Dict[int, float]):
        """Apply mastery predictions to user words"""
        for user_word_id, mastery_level in predictions.items():
            self.user_word_repo.update_mastery_level(user_word_id, mastery_level)

        self.db.commit()

    def update_word_response_time(self, user_word_id: int, response_time: float):
        """Update average response time for a word"""
        self.user_word_repo.update_average_response_time(user_word_id, response_time)
        self.db.commit()
