from typing import Dict

from sqlalchemy.orm import Session

from data.models import User, UserWord, create_tables


class VocabularyLoader:
    def __init__(self):
        self.default_words = [
            # Common words
            {"dutch": "de auto", "english": "car"},
            {"dutch": "de fiets", "english": "bicycle"},
            {"dutch": "het station", "english": "station"},
            # Verbs
            {"dutch": "slapen", "english": "to sleep"},
            # Complex words
            {"dutch": "de regering", "english": "government"},
            {"dutch": "de maatschappij", "english": "society"},
            {"dutch": "de geschiedenis", "english": "history"},
            {"dutch": "de wetenschap", "english": "science"},
            {"dutch": "de ontwikkeling", "english": "development"},
            {"dutch": "de verandering", "english": "change"},
            {"dutch": "de mogelijkheid", "english": "possibility"},
            # Colors
            {"dutch": "rood", "english": "red"},
            # Time
            {"dutch": "de tijd", "english": "time"},
            # Weather
            {"dutch": "het weer", "english": "weather"},
        ]

    def add_word(self, db: Session, dutch_word: str, english_translation: str, user_telegram_id: int = None):
        if not user_telegram_id:
            return False, "User required for adding words"

        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return False, "User not found"

        # Check if user already has this word
        existing_user_word = (
            db.query(UserWord).filter(UserWord.user_id == user.id, UserWord.dutch_word == dutch_word).first()
        )

        if existing_user_word:
            if existing_user_word.is_active:
                return False, "Word already in your vocabulary"
            else:
                # Reactivate the word
                existing_user_word.is_active = True
                db.commit()
                return True, "Word reactivated in your vocabulary"
        else:
            # Add new word to user's vocabulary
            user_word = UserWord(
                user_id=user.id,
                dutch_word=dutch_word,
                english_translation=english_translation,
                word_length=len(dutch_word),
            )
            db.add(user_word)

        db.commit()
        return True, "Word added successfully"

    def add_default_vocabulary_for_user(self, db: Session, user_telegram_id: int):
        """Add default vocabulary to a specific user"""
        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return False, "User not found"

        added_count = 0
        for word_data in self.default_words:
            # Check if user already has this word
            existing_user_word = (
                db.query(UserWord)
                .filter(UserWord.user_id == user.id, UserWord.dutch_word == word_data["dutch"])
                .first()
            )

            if not existing_user_word:
                user_word = UserWord(
                    user_id=user.id,
                    dutch_word=word_data["dutch"],
                    english_translation=word_data["english"],
                    word_length=len(word_data["dutch"]),
                )
                db.add(user_word)
                added_count += 1

        db.commit()
        return True, f"Added {added_count} words to your vocabulary"

    def get_word_stats(self, db: Session, user_telegram_id: int = None) -> Dict:
        if user_telegram_id:
            user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
            if user:
                total_words = db.query(UserWord).filter(UserWord.user_id == user.id).count()
                # Count by word length categories
                from sqlalchemy import func

                length_stats = (
                    db.query(
                        func.avg(UserWord.word_length).label("avg_length"),
                        func.min(UserWord.word_length).label("min_length"),
                        func.max(UserWord.word_length).label("max_length"),
                    )
                    .filter(UserWord.user_id == user.id)
                    .first()
                )
            else:
                return {"error": "User not found"}
        else:
            # Global stats across all users
            total_words = db.query(UserWord).count()
            from sqlalchemy import func

            length_stats = db.query(
                func.avg(UserWord.word_length).label("avg_length"),
                func.min(UserWord.word_length).label("min_length"),
                func.max(UserWord.word_length).label("max_length"),
            ).first()

        return {
            "total_words": total_words,
            "avg_word_length": float(length_stats.avg_length) if length_stats.avg_length else 0,
            "min_word_length": length_stats.min_length or 0,
            "max_word_length": length_stats.max_length or 0,
        }


def initialize_vocabulary():
    create_tables()


if __name__ == "__main__":
    initialize_vocabulary()
