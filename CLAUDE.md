# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Dutch vocabulary learning Telegram bot that uses adaptive machine learning to personalize vocabulary training. The bot implements spaced repetition, contextual bandits for exercise type selection, and learning progress prediction to optimize the learning experience.

## Architecture

### Code Structure

All Python code is organized in the `src/` directory:

**Core Components:**
- **src/bot.py**: Main Telegram bot interface with handlers for user interactions
- **src/exercises.py**: Exercise generation system (multiple choice, translation exercises)  
- **src/spaced_repetition.py**: SM-2 algorithm implementation for optimal review scheduling
- **src/vocabulary_loader.py**: Default vocabulary management and CSV import functionality
- **src/main.py**: Application entry point with setup and initialization

**Data Layer:**
- **src/data/models.py**: SQLAlchemy models and database configuration (User, UserWord, LearningSession, BanditModel)
- **src/data/repositories.py**: Data access layer that decouples ML models from direct database access

**Machine Learning:**
- **src/ml/progress_predictor.py**: `LearningProgressPredictor` - LogisticRegression model for predicting word mastery
- **src/ml/contextual_bandits.py**: `ContextualBandits` - Multi-armed bandit with LogisticRegression for exercise type optimization  
- **src/ml/features.py**: Feature extraction utilities shared between ML models
- **src/ml_models.py**: Backward compatibility module that imports the refactored ML components

### Data Flow

1. Bot receives user input → ExerciseManager generates exercise based on ML recommendations  
2. User response → SpacedRepetitionManager updates review schedule + ML models update via MLDataService
3. ML models continuously adapt to user performance patterns through feature extraction and database repositories

### Architecture Benefits

- **Separation of Concerns**: ML models are decoupled from database access through repositories
- **Code Reusability**: Feature extraction is shared between models to eliminate duplication
- **Testability**: Data access layer makes models easily testable with mock data
- **Maintainability**: Clear module boundaries and single responsibility principle

## Development Commands

### Setup and Running
```bash
# Install dependencies using uv
uv sync

# Setup database and run bot
uv run python src/main.py

# Docker setup
./docker-run.sh
```

### Environment Setup
Create `.env` file with:
- `TELEGRAM_BOT_TOKEN`: Your Telegram bot token
- `DATABASE_URL`: Database connection (defaults to SQLite)

### Database Management
- Database tables are auto-created on first run via `create_tables()`
- Default vocabulary (95 words) loaded automatically on initialization
- SQLite database stored at `dutch_vocab.db` or in Docker volume

## Key Implementation Details

### Machine Learning Integration
- Models retrain automatically after each learning session
- Contextual bandits use epsilon-greedy strategy (ε=0.1) for exploration
- Features include word difficulty, user performance history, response time, time of day
- LearningProgressPredictor: Retrains per user after each session, no file persistence
- ContextualBandits: Model parameters stored in database (`BanditModel` table) per user/exercise type
- New words get prediction of 0 until model is retrained

### Spaced Repetition Logic
- SM-2 algorithm with ease factor adjustment
- Initial intervals: 1 day, 6 days, then calculated based on ease factor
- Incorrect answers reset repetition count and reduce ease factor

### Exercise Types
- `multiple_choice_en_to_nl`: English→Dutch multiple choice
- `multiple_choice_nl_to_en`: Dutch→English multiple choice  
- `translation_en_to_nl`: English→Dutch free text
- `translation_nl_to_en`: Dutch→English free text

### Database Schema Notes
- `User`: Basic user information and Telegram ID
- `UserWord`: Consolidated word data including vocabulary, progress tracking, and spaced repetition
- `LearningSession`: All user interactions with exercises (exercise type, correctness, response time)
- `BanditModel`: ML model parameters for contextual bandits (per user and exercise type)
- Spaced repetition data stored directly in `UserWord` table (next_review_date, repetition_count, ease_factor)