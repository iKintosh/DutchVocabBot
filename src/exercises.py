import random
from typing import Dict, List, Any
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy.orm import Session
from data.models import UserWord

class ExerciseManager:
    def __init__(self):
        self.exercise_types = [
            'multiple_choice_en_to_nl',
            'multiple_choice_nl_to_en', 
            'translation_en_to_nl',
            'translation_nl_to_en'
        ]
    
    def generate_exercise(self, db: Session, user_word: UserWord, exercise_type: str) -> Dict[str, Any]:
        if exercise_type == 'multiple_choice_en_to_nl':
            return self._generate_multiple_choice(db, user_word, 'en_to_nl')
        elif exercise_type == 'multiple_choice_nl_to_en':
            return self._generate_multiple_choice(db, user_word, 'nl_to_en')
        elif exercise_type == 'translation_en_to_nl':
            return self._generate_translation(user_word, 'en_to_nl')
        elif exercise_type == 'translation_nl_to_en':
            return self._generate_translation(user_word, 'nl_to_en')
        else:
            return self._generate_multiple_choice(db, user_word, 'en_to_nl')
    
    def _generate_multiple_choice(self, db: Session, correct_word: UserWord, direction: str) -> Dict[str, Any]:
        # Get 3 random wrong answers from the same user's vocabulary
        wrong_words = db.query(UserWord).filter(
            UserWord.user_id == correct_word.user_id,
            UserWord.id != correct_word.id,
            UserWord.is_active == True
        ).limit(50).all()
        
        if len(wrong_words) < 3:
            # If user doesn't have enough words, fall back to global vocabulary
            # This shouldn't happen in practice since we have 95 default words
            wrong_words = db.query(UserWord).filter(
                UserWord.id != correct_word.id,
                UserWord.is_active == True
            ).limit(50).all()
        
        wrong_options = random.sample(wrong_words, min(3, len(wrong_words)))
        
        if direction == 'en_to_nl':
            question = f"What is the Dutch translation of '{correct_word.english_translation}'?"
            correct_answer = correct_word.dutch_word
            wrong_answers = [w.dutch_word for w in wrong_options]
        else:  # nl_to_en
            question = f"What is the English translation of '{correct_word.dutch_word}'?"
            correct_answer = correct_word.english_translation
            wrong_answers = [w.english_translation for w in wrong_options]
        
        # Create options list and shuffle
        options = [correct_answer] + wrong_answers
        random.shuffle(options)
        
        # Create keyboard
        keyboard = []
        for option in options:
            callback_data = f"exercise_{option}"
            keyboard.append([InlineKeyboardButton(option, callback_data=callback_data)])
        
        return {
            'question': question,
            'keyboard': InlineKeyboardMarkup(keyboard),
            'correct_answer': correct_answer
        }
    
    def _generate_translation(self, word: UserWord, direction: str) -> Dict[str, Any]:
        if direction == 'en_to_nl':
            question = f"Translate to Dutch: '{word.english_translation}'\n\nType your answer:"
            correct_answer = word.dutch_word
        else:  # nl_to_en
            question = f"Translate to English: '{word.dutch_word}'\n\nType your answer:"
            correct_answer = word.english_translation
        
        return {
            'question': question,
            'keyboard': None,  # No keyboard for text input
            'correct_answer': correct_answer
        }
    
    def check_answer(self, word: UserWord, exercise_type: str, user_answer: str) -> bool:
        if exercise_type.endswith('_to_nl'):
            correct_answer = word.dutch_word
        else:
            correct_answer = word.english_translation
        
        # For multiple choice, exact match
        if exercise_type.startswith('multiple_choice'):
            return user_answer.strip().lower() == correct_answer.strip().lower()
        
        # For translation, more flexible matching
        user_clean = user_answer.strip().lower()
        correct_clean = correct_answer.strip().lower()
        
        # Exact match
        if user_clean == correct_clean:
            return True
        
        # Handle articles (de/het) for Dutch
        if exercise_type == 'translation_en_to_nl':
            # Remove articles for comparison
            user_no_article = user_clean.replace('de ', '').replace('het ', '')
            correct_no_article = correct_clean.replace('de ', '').replace('het ', '')
            if user_no_article == correct_no_article:
                return True
        
        # Check if user answer is contained in correct answer or vice versa
        # (to handle cases where correct answer has multiple options)
        if len(user_clean) > 3:  # Avoid matching very short words
            if user_clean in correct_clean or correct_clean in user_clean:
                return True
        
        return False
    
