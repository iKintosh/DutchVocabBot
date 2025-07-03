"""
Feature extraction utilities for machine learning models.

This module provides functions to extract features from user words and learning sessions,
removing duplication between different ML models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

import numpy as np

from data.models import LearningSession, UserWord


@dataclass
class WordFeatures:
    """Word-specific features"""

    length: int
    difficulty: float
    has_article: bool
    is_compound: bool
    has_special_chars: bool
    is_verb: bool
    is_number: bool


@dataclass
class SessionFeatures:
    """Session-specific features"""

    total_sessions: int
    accuracy: float
    avg_response_time: float
    days_since_first: int
    days_since_last: int
    recent_accuracy: float
    exercise_diversity: int


@dataclass
class UserFeatures:
    """User-global features"""

    global_accuracy: float
    hour_of_day: int


def calculate_word_difficulty(word: UserWord) -> float:
    """Calculate word difficulty based on word properties"""
    difficulty = 0.0

    if not word.dutch_word:
        return 0.5

    # Base difficulty from word length (0.1 - 0.5)
    word_length = len(word.dutch_word)
    length_factor = min(0.5, word_length * 0.03)
    difficulty += length_factor

    # Add difficulty for Dutch articles (de/het adds complexity)
    if word.dutch_word.startswith("de ") or word.dutch_word.startswith("het "):
        difficulty += 0.2

    # Add difficulty for compound words (multiple words)
    if len(word.dutch_word.split()) > 1:
        difficulty += 0.15

    # Add difficulty for words with special characters
    special_chars = set("áàäéèëíìïóòöúùüñç")
    if any(char.lower() in special_chars for char in word.dutch_word):
        difficulty += 0.1

    # Infer part of speech and add complexity
    if word.english_translation:
        # Simple heuristics for part of speech inference
        if word.english_translation.startswith("to "):
            difficulty += 0.2  # verb
        elif word.dutch_word.startswith(("de ", "het ", "een ")):
            difficulty += 0.1  # noun
        elif word.english_translation in [
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]:
            difficulty += 0.05  # number
        else:
            difficulty += 0.1  # default (adjective/adverb/etc)

    # Normalize to 0-1 range
    return min(1.0, difficulty)


def extract_word_features(word: UserWord) -> WordFeatures:
    """Extract features from a UserWord object"""
    if not word or not word.dutch_word:
        return WordFeatures(
            length=0,
            difficulty=0.5,
            has_article=False,
            is_compound=False,
            has_special_chars=False,
            is_verb=False,
            is_number=False,
        )

    word_length = len(word.dutch_word)
    word_difficulty = calculate_word_difficulty(word)

    # Additional word features
    has_article = word.dutch_word.startswith("de ") or word.dutch_word.startswith("het ")
    is_compound = len(word.dutch_word.split()) > 1
    has_special_chars = any(char.lower() in "áàäéèëíìïóòöúùüñç" for char in word.dutch_word)

    # Infer word type from patterns
    is_verb = word.english_translation and word.english_translation.startswith("to ")
    is_number = word.english_translation and word.english_translation in [
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
    ]

    return WordFeatures(
        length=word_length,
        difficulty=word_difficulty,
        has_article=has_article,
        is_compound=is_compound,
        has_special_chars=has_special_chars,
        is_verb=is_verb,
        is_number=is_number,
    )


def extract_session_features(sessions: List[LearningSession]) -> SessionFeatures:
    """Extract features from a list of learning sessions"""
    if not sessions:
        return SessionFeatures(
            total_sessions=0,
            accuracy=0,
            avg_response_time=10.0,
            days_since_first=0,
            days_since_last=0,
            recent_accuracy=0,
            exercise_diversity=0,
        )

    total_sessions = len(sessions)
    correct_sessions = sum(1 for s in sessions if s.is_correct)
    accuracy = correct_sessions / total_sessions if total_sessions > 0 else 0

    # Calculate average response time
    response_times = [s.response_time for s in sessions if s.response_time]
    avg_response_time = np.mean(response_times) if response_times else 10.0

    # Calculate time features
    now = datetime.utcnow()
    days_since_first = (now - sessions[0].timestamp).days
    days_since_last = (now - sessions[-1].timestamp).days

    # Recent performance (last 5 sessions)
    recent_sessions = sessions[-5:] if len(sessions) >= 5 else sessions
    recent_accuracy = sum(1 for s in recent_sessions if s.is_correct) / len(recent_sessions)

    # Exercise type diversity
    exercise_types = set(s.exercise_type for s in sessions)
    exercise_diversity = len(exercise_types)

    return SessionFeatures(
        total_sessions=total_sessions,
        accuracy=accuracy,
        avg_response_time=avg_response_time,
        days_since_first=days_since_first,
        days_since_last=days_since_last,
        recent_accuracy=recent_accuracy,
        exercise_diversity=exercise_diversity,
    )


def extract_user_features(user_sessions: List[LearningSession]) -> UserFeatures:
    """Extract user-global features"""
    # Time of day (might affect performance)
    hour_of_day = datetime.now().hour

    # User's overall performance
    if user_sessions:
        correct_user_sessions = [s for s in user_sessions if s.is_correct]
        global_accuracy = len(correct_user_sessions) / len(user_sessions)
    else:
        global_accuracy = 0.5

    return UserFeatures(global_accuracy=global_accuracy, hour_of_day=hour_of_day)


def combine_features_for_progress_prediction(
    word_features: WordFeatures, session_features: SessionFeatures, user_features: UserFeatures
) -> np.ndarray:
    """Combine all features for learning progress prediction"""
    features = [
        word_features.length,
        word_features.difficulty,
        int(word_features.has_article),
        int(word_features.is_compound),
        int(word_features.has_special_chars),
        int(word_features.is_verb),
        int(word_features.is_number),
        session_features.total_sessions,
        session_features.accuracy,
        session_features.avg_response_time,
        session_features.days_since_first,
        session_features.days_since_last,
        session_features.recent_accuracy,
        session_features.exercise_diversity,
        user_features.global_accuracy,
    ]

    return np.array(features).reshape(1, -1)


def combine_features_for_contextual_bandits(
    word_features: WordFeatures, session_features: SessionFeatures, user_features: UserFeatures
) -> np.ndarray:
    """Combine features for contextual bandits (subset of progress features)"""
    features = [
        word_features.length,
        word_features.difficulty,
        int(word_features.has_article),
        int(word_features.is_compound),
        int(word_features.is_verb),
        session_features.accuracy,
        session_features.avg_response_time,
        session_features.total_sessions,
        user_features.hour_of_day,
        user_features.global_accuracy,
    ]

    return np.array(features)
