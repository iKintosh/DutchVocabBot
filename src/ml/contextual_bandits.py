"""
Contextual Bandits for exercise type selection using Logistic Regression.

This module selects optimal exercise types based on user performance patterns.
"""

import json
from datetime import datetime
from typing import Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from data.repositories import MLDataService
from ml.features import (
    combine_features_for_contextual_bandits,
    extract_session_features,
    extract_user_features,
    extract_word_features,
)


class ContextualBandits:
    """Contextual bandits for exercise type selection"""

    def __init__(self):
        self.epsilon = 0.1  # Exploration rate
        self.exercise_types = [
            "multiple_choice_en_to_nl",
            "multiple_choice_nl_to_en",
            "translation_en_to_nl",
            "translation_nl_to_en",
        ]

    def get_context_features(self, data_service: MLDataService, user_id: int, user_word_id: int) -> np.ndarray:
        """Get context features for bandit decision"""
        prediction_data = data_service.get_word_prediction_data(user_id, user_word_id)
        if not prediction_data:
            # Return default features if word not found
            return np.array([0.0] * 10)

        word_features = extract_word_features(prediction_data["user_word"])
        session_features = extract_session_features(prediction_data["word_sessions"])
        user_features = extract_user_features(prediction_data["user_sessions"])

        return combine_features_for_contextual_bandits(word_features, session_features, user_features)

    def select_exercise(self, data_service: MLDataService, user_id: int, user_word_id: int) -> str:
        """Select best exercise type using epsilon-greedy strategy"""
        context = self.get_context_features(data_service, user_id, user_word_id)

        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return np.random.choice(self.exercise_types)

        # Exploit: choose exercise type with highest predicted reward
        best_exercise = None
        best_reward = -np.inf

        for exercise_type in self.exercise_types:
            model_data = data_service.bandit_repo.load_model_data(user_id, exercise_type)
            if model_data and model_data["is_trained"]:
                try:
                    predicted_reward = self._predict_reward(context, model_data)
                    if predicted_reward > best_reward:
                        best_reward = predicted_reward
                        best_exercise = exercise_type
                except Exception as e:
                    print(f"Error predicting for {exercise_type}: {e}")
                    pass

        # If no model is trained or prediction failed, prefer multiple choice exercises
        if best_exercise is None:
            # Bias towards multiple choice exercises when models aren't trained
            weighted_types = (
                ["multiple_choice_en_to_nl"] * 3
                + ["multiple_choice_nl_to_en"] * 3
                + ["translation_en_to_nl"]
                + ["translation_nl_to_en"]
            )
            best_exercise = np.random.choice(weighted_types)

        return best_exercise

    def _predict_reward(self, context: np.ndarray, model_data: Dict) -> float:
        """Predict reward using stored model parameters"""
        # Create and configure model from stored parameters
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.coef_ = np.array(model_data["coefficients"]).reshape(1, -1)
        model.intercept_ = np.array([model_data["intercept"]])
        model.classes_ = np.array([0, 1])  # Binary classification

        # Create scaler from stored parameters
        scaler = StandardScaler()
        scaler.mean_ = np.array(model_data["scaler_mean"])
        scaler.scale_ = np.array(model_data["scaler_scale"])

        # Scale context and predict
        context_scaled = scaler.transform(context.reshape(1, -1))
        predicted_reward = model.predict_proba(context_scaled)[0][1]  # Probability of positive reward

        return predicted_reward

    def update_reward(
        self,
        data_service: MLDataService,
        user_id: int,
        user_word_id: int,
        exercise_type: str,
        is_correct: bool,
        response_time: float,
    ):
        """Update bandit model with reward feedback"""
        context = self.get_context_features(data_service, user_id, user_word_id)

        # Calculate reward based on correctness and response time
        base_reward = 1.0 if is_correct else 0.0  # Binary reward for LogisticRegression
        time_bonus = max(0, (20 - response_time) / 20)  # Bonus for faster responses

        # Binary classification: positive reward (1) vs negative reward (0)
        reward_label = 1 if (base_reward + 0.2 * time_bonus) > 0.5 else 0

        # Get or create model data
        model_data = data_service.bandit_repo.load_model_data(user_id, exercise_type)
        if not model_data:
            model_data = {"contexts": [], "rewards": [], "is_trained": False}
        else:
            # Load existing contexts and rewards (we'll need to maintain them in memory)
            # For now, we'll work with what we have and retrain when we reach threshold
            model_data["contexts"] = model_data.get("contexts", [])
            model_data["rewards"] = model_data.get("rewards", [])

        # Add new context and reward
        model_data["contexts"].append(context.tolist())
        model_data["rewards"].append(reward_label)

        # Retrain model if we have enough data
        if len(model_data["contexts"]) >= 10:  # Lower threshold for per-user models
            self._train_and_save_model(data_service, user_id, exercise_type, model_data)

    def _train_and_save_model(self, data_service: MLDataService, user_id: int, exercise_type: str, model_data: Dict):
        """Train model and save parameters to database"""
        if len(model_data["contexts"]) < 5:
            return

        X = np.array(model_data["contexts"])
        y = np.array(model_data["rewards"])

        try:
            # Create and train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            scaler = StandardScaler()

            # Scale features and train
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)

            # Prepare model data for saving
            save_data = {
                "coefficients": json.dumps(model.coef_[0].tolist()),
                "intercept": float(model.intercept_[0]),
                "scaler_mean": json.dumps(scaler.mean_.tolist()),
                "scaler_scale": json.dumps(scaler.scale_.tolist()),
                "is_trained": True,
                "updated_at": datetime.now(),
            }

            # Save to database
            data_service.bandit_repo.save_model(user_id, exercise_type, save_data)

        except Exception as e:
            print(f"Error training and saving model for {exercise_type}: {e}")

    def get_exercise_performance(self, data_service: MLDataService, user_id: int) -> Dict[str, float]:
        """Get performance statistics for each exercise type"""
        performance = {}

        for exercise_type in self.exercise_types:
            sessions = data_service.session_repo.get_exercise_type_sessions(user_id, exercise_type)

            if sessions:
                correct_sessions = [s for s in sessions if s.is_correct]
                accuracy = len(correct_sessions) / len(sessions)
                performance[exercise_type] = accuracy
            else:
                performance[exercise_type] = 0.0

        return performance
