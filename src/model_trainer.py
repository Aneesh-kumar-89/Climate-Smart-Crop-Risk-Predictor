import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os
from typing import Tuple, Dict, List

class CropRiskModelTrainer:
    """Train and evaluate crop risk prediction models"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boost': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = 0
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for model training"""
        # Create feature columns
        feature_cols = ['temperature', 'rainfall', 'humidity', 'wind_speed', 'soil_moisture']
        
        # Add categorical features
        df_encoded = pd.get_dummies(df, columns=['crop', 'growth_stage'])
        
        # Select features and target
        feature_columns = [col for col in df_encoded.columns if col not in ['risk_level', 'total_stress']]
        X = df_encoded[feature_columns]
        y = df['risk_level']
        
        return X, y, feature_columns
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'train_score': train_score,
                'test_score': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Update best model
            if test_score > self.best_score:
                self.best_score = test_score
                self.best_model = model
        
        return results
    
    def save_model(self, model_path: str = 'models/crop_risk_model.pkl', 
                   scaler_path: str = 'models/scaler.pkl'):
        """Save trained model and scaler"""
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load_model(self, model_path: str = 'models/crop_risk_model.pkl', 
                   scaler_path: str = 'models/scaler.pkl'):
        """Load trained model and scaler"""
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        print("Model and scaler loaded successfully")
        
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance_df
        return pd.DataFrame()