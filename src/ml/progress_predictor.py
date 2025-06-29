"""
Learning Progress Predictor using Logistic Regression.

This module predicts word mastery levels based on user performance patterns.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, Optional

from data.repositories import MLDataService
from ml.features import (
    extract_word_features, 
    extract_session_features, 
    extract_user_features,
    combine_features_for_progress_prediction
)


class LearningProgressPredictor:
    """Predicts learning progress using logistic regression"""
    
    def __init__(self):
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def extract_features(self, data_service: MLDataService, user_id: int, user_word_id: int) -> Optional[np.ndarray]:
        """Extract features for a single word prediction"""
        prediction_data = data_service.get_word_prediction_data(user_id, user_word_id)
        if not prediction_data:
            return None
        
        word_features = extract_word_features(prediction_data['user_word'])
        session_features = extract_session_features(prediction_data['word_sessions'])
        user_features = extract_user_features(prediction_data['user_sessions'])
        
        return combine_features_for_progress_prediction(word_features, session_features, user_features)
    
    def train_model(self, data_service: MLDataService, user_id: Optional[int] = None) -> bool:
        """Train the model on user data"""
        training_data = data_service.get_word_training_data(user_id)
        
        if len(training_data) < 5:  # Lower threshold since we train per user
            return False
        
        features_list = []
        targets = []
        
        for data_point in training_data:
            word_features = extract_word_features(data_point['user_word'])
            session_features = extract_session_features(data_point['word_sessions'])
            user_features = extract_user_features(data_point['user_sessions'])
            
            features = combine_features_for_progress_prediction(word_features, session_features, user_features)
            features_list.append(features.flatten())
            targets.append(data_point['target'])
        
        X = np.array(features_list)
        y = np.array(targets)
        
        # Check if we have both classes (0 and 1) for binary classification
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return False  # Cannot train with only one class
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        return True
    
    def predict_mastery(self, data_service: MLDataService, user_id: int, user_word_id: int) -> float:
        """Predict mastery level for a specific word"""
        prediction_data = data_service.get_word_prediction_data(user_id, user_word_id)
        if not prediction_data:
            return 0.0
        
        # Check if user word has been seen
        if prediction_data['user_word'].times_seen == 0:
            return 0.0  # New words get 0 prediction
        
        if not self.is_trained:
            return 0.0  # Default for untrained model
        
        features = self.extract_features(data_service, user_id, user_word_id)
        if features is None:
            return 0.0
        
        features_scaled = self.scaler.transform(features)
        
        # Get probability of being mastered (class 1)
        prediction_proba = self.model.predict_proba(features_scaled)[0][1]
        return prediction_proba
    
    def update_progress_and_retrain(self, data_service: MLDataService, user_id: int, user_word_id: int, response_time: Optional[float]):
        """Update user word progress and retrain model"""
        # Update average response time if provided
        if response_time:
            data_service.update_word_response_time(user_word_id, response_time)
        
        # Retrain model after every session for this user
        self.train_model(data_service, user_id)
        
        # Apply predictions to all user words
        self.apply_predictions_to_user_words(data_service, user_id)
    
    def apply_predictions_to_user_words(self, data_service: MLDataService, user_id: int):
        """Apply ML predictions to all user words after training"""
        if not self.is_trained:
            return
        
        user_words = data_service.user_word_repo.get_all_user_words(user_id)
        predictions = {}
        
        for user_word in user_words:
            if user_word.times_seen > 0:
                # Get ML prediction for this word
                predicted_mastery = self.predict_mastery(data_service, user_id, user_word.id)
                predictions[user_word.id] = predicted_mastery
        
        # Apply all predictions in batch
        data_service.apply_mastery_predictions(user_id, predictions)