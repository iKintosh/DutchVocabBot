import csv
from sqlalchemy.orm import Session
from data.models import UserWord, User, get_db, create_tables
from typing import List, Dict

class VocabularyLoader:
    def __init__(self):
        self.default_words = [
            # Basic words
            {"dutch": "hallo", "english": "hello"},
            {"dutch": "dank je", "english": "thank you"},
            {"dutch": "ja", "english": "yes"},
            {"dutch": "nee", "english": "no"},
            {"dutch": "goed", "english": "good"},
            {"dutch": "slecht", "english": "bad"},
            {"dutch": "water", "english": "water"},
            {"dutch": "het brood", "english": "bread"},
            {"dutch": "de melk", "english": "milk"},
            {"dutch": "het huis", "english": "house"},
            
            # Common words
            {"dutch": "de auto", "english": "car"},
            {"dutch": "de fiets", "english": "bicycle"},
            {"dutch": "het station", "english": "station"},
            {"dutch": "de winkel", "english": "shop"},
            {"dutch": "het restaurant", "english": "restaurant"},
            {"dutch": "de school", "english": "school"},
            {"dutch": "het werk", "english": "work"},
            {"dutch": "de familie", "english": "family"},
            {"dutch": "de vriend", "english": "friend"},
            {"dutch": "het eten", "english": "food"},
            
            # Verbs
            {"dutch": "zijn", "english": "to be"},
            {"dutch": "hebben", "english": "to have"},
            {"dutch": "gaan", "english": "to go"},
            {"dutch": "komen", "english": "to come"},
            {"dutch": "doen", "english": "to do"},
            {"dutch": "zien", "english": "to see"},
            {"dutch": "horen", "english": "to hear"},
            {"dutch": "eten", "english": "to eat"},
            {"dutch": "drinken", "english": "to drink"},
            {"dutch": "slapen", "english": "to sleep"},
            
            # Complex words
            {"dutch": "de regering", "english": "government"},
            {"dutch": "de maatschappij", "english": "society"},
            {"dutch": "de economie", "english": "economy"},
            {"dutch": "de cultuur", "english": "culture"},
            {"dutch": "de geschiedenis", "english": "history"},
            {"dutch": "de wetenschap", "english": "science"},
            {"dutch": "de technologie", "english": "technology"},
            {"dutch": "de ontwikkeling", "english": "development"},
            {"dutch": "de verandering", "english": "change"},
            {"dutch": "de mogelijkheid", "english": "possibility"},
            
            # Colors
            {"dutch": "rood", "english": "red"},
            {"dutch": "blauw", "english": "blue"},
            {"dutch": "groen", "english": "green"},
            {"dutch": "geel", "english": "yellow"},
            {"dutch": "zwart", "english": "black"},
            {"dutch": "wit", "english": "white"},
            
            # Numbers
            {"dutch": "een", "english": "one"},
            {"dutch": "twee", "english": "two"},
            {"dutch": "drie", "english": "three"},
            {"dutch": "vier", "english": "four"},
            {"dutch": "vijf", "english": "five"},
            {"dutch": "zes", "english": "six"},
            {"dutch": "zeven", "english": "seven"},
            {"dutch": "acht", "english": "eight"},
            {"dutch": "negen", "english": "nine"},
            {"dutch": "tien", "english": "ten"},
            
            # Time
            {"dutch": "de tijd", "english": "time"},
            {"dutch": "de dag", "english": "day"},
            {"dutch": "de week", "english": "week"},
            {"dutch": "de maand", "english": "month"},
            {"dutch": "het jaar", "english": "year"},
            {"dutch": "vandaag", "english": "today"},
            {"dutch": "gisteren", "english": "yesterday"},
            {"dutch": "morgen", "english": "tomorrow"},
            
            # Weather
            {"dutch": "het weer", "english": "weather"},
            {"dutch": "de zon", "english": "sun"},
            {"dutch": "de regen", "english": "rain"},
            {"dutch": "de wind", "english": "wind"},
            {"dutch": "de sneeuw", "english": "snow"},
            {"dutch": "warm", "english": "warm"},
            {"dutch": "koud", "english": "cold"},
        ]
    
    def load_default_vocabulary(self, db: Session):
        # This method is now deprecated since each user has their own vocabulary
        # Use add_default_vocabulary_for_user instead
        print("load_default_vocabulary is deprecated. Use add_default_vocabulary_for_user instead.")
    
    def load_from_csv(self, db: Session, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                loaded_count = 0
                
                for row in reader:
                    # This method needs to be updated to work with UserWord
                    # For now, raise an error
                    raise NotImplementedError("CSV loading needs to be updated for UserWord model")
                
                db.commit()
                print(f"Loaded {loaded_count} words from CSV file")
                
        except FileNotFoundError:
            print(f"CSV file not found: {file_path}")
        except Exception as e:
            print(f"Error loading CSV file: {e}")
    
    def add_word(self, db: Session, dutch_word: str, english_translation: str, 
                 user_telegram_id: int = None):
        if not user_telegram_id:
            return False, "User required for adding words"
            
        user = db.query(User).filter(User.telegram_id == user_telegram_id).first()
        if not user:
            return False, "User not found"
        
        # Check if user already has this word
        existing_user_word = db.query(UserWord).filter(
            UserWord.user_id == user.id,
            UserWord.dutch_word == dutch_word
        ).first()
        
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
                word_length=len(dutch_word)
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
            existing_user_word = db.query(UserWord).filter(
                UserWord.user_id == user.id,
                UserWord.dutch_word == word_data["dutch"]
            ).first()
            
            if not existing_user_word:
                user_word = UserWord(
                    user_id=user.id,
                    dutch_word=word_data["dutch"],
                    english_translation=word_data["english"],
                    word_length=len(word_data["dutch"])
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
                length_stats = db.query(
                    func.avg(UserWord.word_length).label('avg_length'),
                    func.min(UserWord.word_length).label('min_length'),
                    func.max(UserWord.word_length).label('max_length')
                ).filter(UserWord.user_id == user.id).first()
            else:
                return {"error": "User not found"}
        else:
            # Global stats across all users
            total_words = db.query(UserWord).count()
            from sqlalchemy import func
            length_stats = db.query(
                func.avg(UserWord.word_length).label('avg_length'),
                func.min(UserWord.word_length).label('min_length'),
                func.max(UserWord.word_length).label('max_length')
            ).first()
        
        return {
            "total_words": total_words,
            "avg_word_length": float(length_stats.avg_length) if length_stats.avg_length else 0,
            "min_word_length": length_stats.min_length or 0,
            "max_word_length": length_stats.max_length or 0
        }

def initialize_vocabulary():
    create_tables()
    loader = VocabularyLoader()
    
    with next(get_db()) as db:
        loader.load_default_vocabulary(db)
        stats = loader.get_word_stats(db)
        print("Vocabulary initialization complete:")
        print(f"Total words: {stats['total_words']}")
        print(f"Average word length: {stats['avg_word_length']:.1f} characters")

if __name__ == "__main__":
    initialize_vocabulary()