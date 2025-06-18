import logging
import os
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
from sqlalchemy.orm import Session
from data.models import get_db, create_tables, User, UserWord, LearningSession
from data.repositories import MLDataService
from exercises import ExerciseManager
from spaced_repetition import SpacedRepetitionManager
from ml_models import LearningProgressPredictor, ContextualBandits
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

class DutchVocabBot:
    """
    Main Telegram bot class for Dutch vocabulary training.
    
    This bot uses machine learning and spaced repetition to optimize vocabulary learning:
    - ExerciseManager: Generates different types of exercises (multiple choice, translation)
    - SpacedRepetitionManager: Implements SM-2 algorithm for optimal review scheduling
    - LearningProgressPredictor: ML model that predicts word mastery levels
    - ContextualBandits: ML algorithm that selects optimal exercise types for each word
    """
    def __init__(self):
        self.exercise_manager = ExerciseManager()
        self.sr_manager = SpacedRepetitionManager()
        self.progress_predictor = LearningProgressPredictor()
        self.bandits = ContextualBandits()
        
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /start command - welcome new users and show main menu.
        
        For new users:
        1. Creates user record in database
        2. Adds default vocabulary (95 Dutch words)
        
        Shows main menu with options:
        - Start Learning: Begin vocabulary exercises
        - View Progress: Show learning statistics
        - Add Word: Add custom vocabulary
        - Settings: Bot configuration (placeholder)
        """
        user = update.effective_user
        if not user or not update.message:
            return
        
        with next(get_db()) as db:
            # Check if user exists, create if new
            db_user = db.query(User).filter(User.telegram_id == user.id).first()
            if not db_user:
                db_user = User(
                    telegram_id=user.id,
                    username=user.username,
                    first_name=user.first_name
                )
                db.add(db_user)
                db.commit()
                
                # Add default vocabulary for new user (95 Dutch words)
                from vocabulary_loader import VocabularyLoader
                loader = VocabularyLoader()
                success, message = loader.add_default_vocabulary_for_user(db, user.id)
                if success:
                    print(f"Added default vocabulary for user {user.id}: {message}")
        
        # Create main menu keyboard
        keyboard = [
            [InlineKeyboardButton("Start Learning", callback_data="start_learning")],
            [InlineKeyboardButton("View Progress", callback_data="view_progress")],
            [InlineKeyboardButton("Add Word", callback_data="add_word_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            f"Welcome to Dutch Vocabulary Trainer, {user.username}! 🇳🇱\n\n"
            "I'll help you learn Dutch words using adaptive exercises and spaced repetition.",
            reply_markup=reply_markup
        )
    
    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        if not query or not query.data:
            return
            
        await query.answer()
        
        if query.data == "start_learning":
            await self.start_learning_session(query, context)
        elif query.data == "view_progress":
            await self.show_progress(query, context)
        elif query.data == "add_word_menu":
            await self.show_add_word_menu(query, context)
        elif query.data.startswith("exercise_"):
            await self.handle_exercise_response(query, context)
        elif query.data == "next_word":
            await self.start_learning_session(query, context)
        elif query.data == "finish_session":
            await self.finish_learning_session(query, context)
        elif query.data == "back_to_menu":
            await self.show_main_menu(query, context)
    
    async def start_learning_session(self, query, context):
        """
        Start a new learning session with adaptive word selection and exercise type optimization.
        
        Learning Session Flow:
        1. Use SpacedRepetitionManager to select next word (due words prioritized)
        2. Use ContextualBandits ML to select optimal exercise type for the word
        3. Generate exercise (multiple choice or translation)
        4. Track session progress (max 10 words per session)
        5. Store exercise context for response handling
        
        The system adapts to user performance:
        - Words are scheduled using SM-2 spaced repetition algorithm
        - Exercise types are selected based on user's historical performance
        - ML models predict word mastery and optimize learning
        """
        if not query.from_user:
            return
            
        user_id = query.from_user.id
        
        with next(get_db()) as db:
            # Get next word to review based on spaced repetition and ML predictions
            next_word = self.sr_manager.get_next_word_for_review(db, user_id, self.progress_predictor)
            
            if not next_word:
                await query.edit_message_text("No words to review right now! Check back later.")
                return
            
            # Get best exercise type using contextual bandits
            data_service = MLDataService(db)
            exercise_type = self.bandits.select_exercise(data_service, user_id, next_word.id)
            
            # Generate exercise
            exercise_data = self.exercise_manager.generate_exercise(db, next_word, exercise_type)
            
            # Store current exercise in context and track session
            context.user_data['current_word_id'] = next_word.id
            context.user_data['current_exercise_type'] = exercise_type
            context.user_data['exercise_start_time'] = datetime.now()
            
            # Initialize or increment session counter
            session_count = context.user_data.get('session_count', 0) + 1
            context.user_data['session_count'] = session_count
            
            # Check if session limit reached
            if session_count > 9:
                await self.finish_learning_session(query, context)
                return
            
            # Add finish session button to keyboard if it's not a translation exercise
            if exercise_data['keyboard'] and session_count >= 1:
                finish_button = InlineKeyboardButton("Finish Session", callback_data="finish_session")
                # Create new keyboard with existing buttons plus finish button
                existing_buttons = exercise_data['keyboard'].inline_keyboard
                new_keyboard = list(existing_buttons) + [[finish_button]]
                exercise_data['keyboard'] = InlineKeyboardMarkup(new_keyboard)
            
            session_text = f"Word {session_count}/10\n\n{exercise_data['question']}"
            
            await query.edit_message_text(
                session_text,
                reply_markup=exercise_data['keyboard']
            )
    
    async def handle_exercise_response(self, query, context):
        if not query.from_user:
            return
            
        user_id = query.from_user.id
        word_id = context.user_data.get('current_word_id')
        exercise_type = context.user_data.get('current_exercise_type')
        start_time = context.user_data.get('exercise_start_time')
        
        if not all([word_id, exercise_type, start_time]):
            await query.edit_message_text("Session expired. Please start again.")
            return
        
        if not isinstance(word_id, int) or not isinstance(exercise_type, str):
            await query.edit_message_text("Invalid session data. Please start again.")
            return
        
        if not isinstance(start_time, datetime):
            await query.edit_message_text("Invalid session timing. Please start again.")
            return
            
        response_time = (datetime.now() - start_time).total_seconds()
        user_answer = query.data.replace("exercise_", "") if query.data else ""
        
        with next(get_db()) as db:
            user_word = db.query(UserWord).filter(UserWord.id == word_id).first()
            if not user_word:
                await query.edit_message_text("Word not found. Please start again.")
                return
                
            is_correct = self.exercise_manager.check_answer(user_word, exercise_type, user_answer)
            
            user_db = db.query(User).filter(User.telegram_id == user_id).first()
            if not user_db:
                await query.edit_message_text("User not found. Please start again.")
                return
            
            # Record learning session
            session = LearningSession(
                user_id=user_db.id,
                user_word_id=word_id,
                exercise_type=exercise_type,
                is_correct=is_correct,
                response_time=response_time,
                timestamp=datetime.now()
            )
            db.add(session)
            
            # Update spaced repetition schedule (word_id is actually user_word_id)
            self.sr_manager.update_word_schedule(db, user_id, word_id, is_correct)
            
            # Update ML models (word_id is actually user_word_id)
            data_service = MLDataService(db)
            self.progress_predictor.update_progress_and_retrain(data_service, user_id, word_id, session.response_time)
            self.bandits.update_reward(data_service, user_id, word_id, exercise_type, is_correct, response_time)
            
            db.commit()
            
            # Show result
            correct_answer = user_word.english_translation if exercise_type.endswith('_to_en') else user_word.dutch_word
            result_text = "✅ Correct!" if is_correct else f"❌ Incorrect. The answer was: {correct_answer}"
            
            keyboard = [
                [InlineKeyboardButton("Next Word", callback_data="next_word")], 
                [InlineKeyboardButton("Finish Session", callback_data="finish_session")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(result_text, reply_markup=reply_markup)
    
    async def show_progress(self, query, context):
        keyboard = [
                [InlineKeyboardButton("Start Learning", callback_data="start_learning")],
                [InlineKeyboardButton("Main Menu", callback_data="back_to_menu")]
            ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        if not query.from_user:
            return
            
        user_id = query.from_user.id
        
        with next(get_db()) as db:
            user_db = db.query(User).filter(User.telegram_id == user_id).first()
            if not user_db:
                await query.edit_message_text("User not found. Please start again.", reply_markup=reply_markup)
                return
                
            sessions = db.query(LearningSession).filter(LearningSession.user_id == user_db.id).all()

            words = db.query(UserWord).filter(UserWord.user_id == user_db.id).all()
            
            if not sessions:
                await query.edit_message_text("No learning sessions yet. Start learning to see your progress!", reply_markup=reply_markup)
                return
            
            total_sessions = len(sessions)
            correct_answers = sum(1 for s in sessions if s.is_correct)
            accuracy = (correct_answers / total_sessions) * 100 if total_sessions > 0 else 0
            avg_mastery_level = sum(w.mastery_level for w in words) / len(words)
            
            progress_text = f"📊 Your Progress:\n\n"
            progress_text += f"Total exercises: {total_sessions}\n"
            progress_text += f"Correct answers: {correct_answers}\n"
            progress_text += f"Accuracy: {accuracy:.1f}%\n"
            progress_text += f"Average Mastery: {avg_mastery_level:.1f}%\n"
            
            await query.edit_message_text(progress_text, reply_markup=reply_markup)
    
    async def handle_text_input(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle text input for translation exercises.
        
        Translation exercises require text input rather than button clicks.
        This method:
        1. Validates the user has an active translation exercise
        2. Checks the user's answer against the correct translation
        3. Records the learning session and updates ML models
        4. Updates spaced repetition schedule based on correctness
        """
        if not update.effective_user or not update.message or not update.message.text:
            return
            
        user_id = update.effective_user.id
        user_answer = update.message.text.strip()
        
        # Check if user is currently in a translation exercise
        if not context.user_data:
            await update.message.reply_text("No active exercise. Use /start to begin learning!")
            return
            
        current_word_id = context.user_data.get('current_word_id')
        exercise_type = context.user_data.get('current_exercise_type')
        start_time = context.user_data.get('exercise_start_time')
        
        if not all([current_word_id, exercise_type, start_time]):
            await update.message.reply_text("No active exercise. Use /start to begin learning!")
            return
        
        if not isinstance(current_word_id, int) or not isinstance(exercise_type, str):
            await update.message.reply_text("Invalid session data. Please start again.")
            return
        
        if not exercise_type.startswith('translation_'):
            await update.message.reply_text("Please use the buttons for multiple choice questions.")
            return
        
        if not isinstance(start_time, datetime):
            await update.message.reply_text("Invalid session state. Please start again.")
            return
            
        response_time = (datetime.now() - start_time).total_seconds()
        
        with next(get_db()) as db:
            user_word = db.query(UserWord).filter(UserWord.id == current_word_id).first()
            if not user_word:
                await update.message.reply_text("Word not found. Please start again.")
                return
                
            is_correct = self.exercise_manager.check_answer(user_word, exercise_type, user_answer)
            
            user_db = db.query(User).filter(User.telegram_id == user_id).first()
            if not user_db:
                await update.message.reply_text("User not found. Please start again.")
                return
            
            # Record learning session
            session = LearningSession(
                user_id=user_db.id,
                user_word_id=current_word_id,
                exercise_type=exercise_type,
                is_correct=is_correct,
                response_time=response_time,
                timestamp=datetime.now()
            )
            db.add(session)
            
            # Update spaced repetition schedule
            self.sr_manager.update_word_schedule(db, user_id, current_word_id, is_correct)
            
            # Update ML models
            data_service = MLDataService(db)
            self.progress_predictor.update_progress_and_retrain(data_service, user_id, current_word_id, session.response_time)
            self.bandits.update_reward(data_service, user_id, current_word_id, exercise_type, is_correct, response_time)
            
            db.commit()
            
            # Clear current exercise from context
            if context.user_data:
                context.user_data.pop('current_word_id', None)
                context.user_data.pop('current_exercise_type', None)
                context.user_data.pop('exercise_start_time', None)
            
            # Show result
            correct_answer = user_word.english_translation if exercise_type.endswith('_to_en') else user_word.dutch_word
            result_text = "✅ Correct!" if is_correct else f"❌ Incorrect. The answer was: {correct_answer}"
            
            keyboard = [[InlineKeyboardButton("Next Word", callback_data="next_word")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(result_text, reply_markup=reply_markup)
    
    async def add_word_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add_word command to add custom vocabulary"""
        if not update.message:
            return
            
        if not context.args or len(context.args) < 2:
            await update.message.reply_text(
                "Usage: /add_word \"<dutch_word>\" \"<english_translation>\"\n"
                "Example: /add_word \"de kat\" \"cat\""
            )
            return
        
        dutch_word = context.args[0]
        english_translation = context.args[1]
        
        from vocabulary_loader import VocabularyLoader
        loader = VocabularyLoader()
        
        if not update.effective_user:
            return
            
        with next(get_db()) as db:
            success, message = loader.add_word(db, dutch_word, english_translation, user_telegram_id=update.effective_user.id)
            await update.message.reply_text(f"{'✅' if success else '❌'} {message}")
    
    async def show_add_word_menu(self, query, context):
        """Show add word menu with instructions"""
        await query.edit_message_text(
            "Add a new word to your vocabulary:\n\n"
            "Use the command: /add_word <dutch_word> <english_translation>\n\n"
            "Examples:\n"
            "• /add_word \"de kat\" \"cat\"\n"
            "• /add_word \"lopen\" \"to walk\"\n"
            "• /add_word \"mooi\" \"beautiful\"\n\n"
            "The word will be added to your personal vocabulary and you can start practicing it immediately!",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Back to Menu", callback_data="back_to_menu")]])
        )
    
    async def finish_learning_session(self, query, context):
        """Finish the current learning session"""
        session_count = context.user_data.get('session_count', 0)
        
        # Clear session data
        if context.user_data:
            context.user_data.pop('current_word_id', None)
            context.user_data.pop('current_exercise_type', None)
            context.user_data.pop('exercise_start_time', None)
            context.user_data.pop('session_count', None)
        
        session_text = f"🎉 Session completed! You practiced {session_count} words.\n\nGreat work! Regular practice helps with long-term retention."
        
        keyboard = [
            [InlineKeyboardButton("Start New Session", callback_data="start_learning")],
            [InlineKeyboardButton("View Progress", callback_data="view_progress")],
            [InlineKeyboardButton("Main Menu", callback_data="back_to_menu")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(session_text, reply_markup=reply_markup)
    
    async def show_main_menu(self, query, context):
        """Show the main menu"""
        keyboard = [
            [InlineKeyboardButton("Start Learning", callback_data="start_learning")],
            [InlineKeyboardButton("View Progress", callback_data="view_progress")],
            [InlineKeyboardButton("Add Word", callback_data="add_word_menu")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(
            "I'll help you learn Dutch words using adaptive exercises and spaced repetition.",
            reply_markup=reply_markup
        )

def main():
    create_tables()
    
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_BOT_TOKEN environment variable is required")
    
    application = Application.builder().token(token).build()
    
    bot = DutchVocabBot()
    
    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(CommandHandler("add_word", bot.add_word_command))
    application.add_handler(CallbackQueryHandler(bot.button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_text_input))
    
    application.run_polling()

if __name__ == '__main__':
    main()