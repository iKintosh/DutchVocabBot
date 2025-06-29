from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_
from data.models import User, UserWord, LearningSession

class SpacedRepetitionManager:
    def __init__(self):
        # SM-2 algorithm parameters
        self.default_ease_factor = 2.5
        self.min_ease_factor = 1.3
        self.ease_factor_bonus = 0.1
        self.ease_factor_penalty = 0.2
        
        # Initial intervals (in days)
        self.initial_intervals = [1, 6]
    
    def get_next_word_for_review(self, db: Session, user_telegram_id: int, progress_predictor=None) -> Optional[UserWord]:
        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return None
        due_words: list[UserWord] = self._get_due_words(db, user.id)
        if due_words: 
            return due_words[0]
        return None
    
    def _get_due_words(self, db: Session, user_id: int) -> list[UserWord]:
        
        # Get user words that are due for review (including those with null review dates)
        due_words = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.is_active == True,
            )
        ).order_by(UserWord.next_review_date.nulls_first(), UserWord.mastery_level.nulls_first()).all()
        
        return due_words
    
    def update_word_schedule(self, db: Session, user_telegram_id: int, user_word_id: int, is_correct: bool):
        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return
        
        # Get the user word
        user_word = db.query(UserWord).filter(UserWord.id == user_word_id).first()
        if not user_word:
            return
        
        # Calculate new schedule using SM-2 algorithm
        repetition_count = user_word.repetition_count + 1
        ease_factor = user_word.ease_factor
        
        if is_correct:
            # Correct answer - increase interval
            if repetition_count == 1:
                interval_days = self.initial_intervals[0]
            elif repetition_count == 2:
                interval_days = self.initial_intervals[1]
            else:
                # Calculate interval based on previous interval and ease factor
                prev_interval = self._get_previous_interval(db, user.id, user_word_id)
                interval_days = max(1, int(prev_interval * ease_factor))
            
            # Adjust ease factor for correct answer
            ease_factor = min(3.0, ease_factor + self.ease_factor_bonus)
            
        else:
            # Incorrect answer - reset to beginning
            repetition_count = 0
            interval_days = 1
            ease_factor = max(self.min_ease_factor, ease_factor - self.ease_factor_penalty)
        
        # Update the user word with new schedule
        next_review_date = datetime.utcnow() + timedelta(days=interval_days)
        user_word.next_review_date = next_review_date
        user_word.repetition_count = repetition_count
        user_word.ease_factor = ease_factor
        
        # Update user word progress
        self._update_user_word_progress(db, user_word, is_correct)
        
        db.commit()
    
    def _get_previous_interval(self, db: Session, user_id: int, user_word_id: int) -> int:
        # Get the two most recent sessions to calculate previous interval
        sessions = db.query(LearningSession).filter(
            and_(
                LearningSession.user_id == user_id,
                LearningSession.user_word_id == user_word_id
            )
        ).order_by(LearningSession.timestamp.desc()).limit(2).all()
        
        if len(sessions) < 2:
            return 1
        
        time_diff = sessions[0].timestamp - sessions[1].timestamp
        return max(1, time_diff.days)
    
    def _update_user_word_progress(self, db: Session, user_word: UserWord, is_correct: bool):
        # Update progress tracking fields
        user_word.times_seen += 1
        if is_correct:
            user_word.times_correct += 1
        user_word.last_seen = datetime.utcnow()
        
        # Calculate new mastery level
        accuracy = user_word.times_correct / user_word.times_seen if user_word.times_seen > 0 else 0
        user_word.mastery_level = min(1.0, accuracy * (user_word.times_seen / 10))  # Scale by frequency
    
    def get_review_stats(self, db: Session, user_telegram_id: int) -> dict:
        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return {}
        
        now = datetime.utcnow()
        
        # Count due words
        due_count = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user.id,
                UserWord.is_active == True,
                UserWord.next_review_date <= now
            )
        ).count()
        
        # Count new words available
        new_count = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user.id,
                UserWord.is_active == True,
                UserWord.times_seen == 0
            )
        ).count()
        
        # Count total words in progress
        total_learning = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user.id,
                UserWord.is_active == True,
                UserWord.times_seen > 0
            )
        ).count()
        
        return {
            'due_for_review': due_count,
            'new_words_available': new_count,
            'total_words_learning': total_learning
        }