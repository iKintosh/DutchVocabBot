from datetime import datetime, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
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
        
        # First, try to get words that are due for review
        due_words = self._get_due_words(db, user.id)
        if due_words:
            # If we have a progress predictor, prioritize words with lowest mastery
            if progress_predictor and len(due_words) > 1:
                return self._prioritize_by_mastery(db, user.id, due_words, progress_predictor)
            return due_words[0]
        
        # If no due words, get a new word that hasn't been seen
        new_word = self._get_new_word(db, user.id)
        if new_word:
            return new_word
        
        # If no new words, get words that need the most practice (lowest mastery)
        if progress_predictor:
            return self._get_lowest_mastery_word(db, user.id, progress_predictor)
        
        # Fallback: get the least recently seen word
        return self._get_least_recent_word(db, user.id)
    
    def _prioritize_by_mastery(self, db: Session, user_id: int, words: list[UserWord], progress_predictor) -> UserWord:
        """Select word with lowest predicted mastery from the given list"""
        word_masteries = []
        for word in words:
            mastery = progress_predictor.predict_mastery(db, user_id, word.id)
            word_masteries.append((word, mastery))
        
        # Sort by mastery (lowest first) and return the least mastered word
        word_masteries.sort(key=lambda x: x[1])
        return word_masteries[0][0]
    
    def _get_lowest_mastery_word(self, db: Session, user_id: int, progress_predictor) -> Optional[UserWord]:
        """Get the word from user's vocabulary with lowest predicted mastery"""
        # Get all words from user's vocabulary that have been seen at least once
        candidate_words = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.is_active == True,
                UserWord.times_seen > 0
            )
        ).all()
        
        if not candidate_words:
            return None
        
        # Find word with lowest mastery
        lowest_mastery = float('inf')
        best_word = None
        
        for user_word in candidate_words:
            mastery = progress_predictor.predict_mastery(db, user_id, user_word.id)
            if mastery < lowest_mastery:
                lowest_mastery = mastery
                best_word = user_word
        
        return best_word
    
    def _get_due_words(self, db: Session, user_id: int) -> list[UserWord]:
        now = datetime.utcnow()
        
        # Get user words that are due for review (including those with null review dates)
        due_words = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.is_active == True,
                or_(
                    UserWord.next_review_date <= now,
                    UserWord.next_review_date.is_(None)
                )
            )
        ).order_by(UserWord.next_review_date.nulls_first()).all()
        
        return due_words
    
    def _get_new_word(self, db: Session, user_id: int) -> Optional[UserWord]:
        # Get words from user's vocabulary that they haven't seen yet
        new_word = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.is_active == True,
                UserWord.times_seen == 0
            )
        ).order_by(UserWord.id).first()
        
        return new_word
    
    def _get_least_recent_word(self, db: Session, user_id: int) -> Optional[UserWord]:
        # Get the user word that was seen least recently
        least_recent_word = db.query(UserWord).filter(
            and_(
                UserWord.user_id == user_id,
                UserWord.is_active == True,
                UserWord.times_seen > 0
            )
        ).order_by(UserWord.last_seen.asc()).first()
        
        return least_recent_word
    
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