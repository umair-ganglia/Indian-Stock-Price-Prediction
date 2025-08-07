"""
Ensemble Models for Enhanced Stock Prediction
Author: [Your Name]
Date: 2025

Advanced ensemble methods combining multiple ML algorithms
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

class EnsemblePredictor:
    """Advanced ensemble prediction methods"""
    
    def __init__(self):
        self.ensemble_model = None
        self.stacked_model = None
        self.models = {}
        self.predictions = {}
    
    def create_voting_ensemble(self):
        """Create voting ensemble of different algorithms"""
        
        # Base models
        lr = LinearRegression()
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        estimators = [
            ('lr', lr),
            ('rf', rf), 
            ('gb', gb)
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb = XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            estimators.append(('xgb', xgb))
        
        # Create ensemble
        self.ensemble_model = VotingRegressor(estimators)
        
        return self.ensemble_model
    
    def create_stacking_ensemble(self):
        """Create stacked ensemble model"""
        
        # Base models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', XGBRegressor(n_estimators=50, random_state=42, n_jobs=-1)))
        
        # Meta-model
        meta_model = LinearRegression()
        
        # Create stacking regressor
        self.stacked_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
        
        return self.stacked_model
    
    def train_ensemble(self, X_train, y_train, ensemble_type='voting'):
        """Train ensemble model"""
        
        if ensemble_type == 'voting':
            model = self.create_voting_ensemble()
        elif ensemble_type == 'stacking':
            model = self.create_stacking_ensemble()
        else:
            raise ValueError("ensemble_type must be 'voting' or 'stacking'")
        
        print(f"🤖 Training {ensemble_type} ensemble...")
        model.fit(X_train, y_train)
        
        return model
    
    def evaluate_ensemble(self, model, X_test, y_test, model_name):
        """Evaluate ensemble model"""
        
        predictions = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        results = {
            'RMSE': rmse,
            'R²': r2,
            'predictions': predictions
        }
        
        self.predictions[model_name] = results
        
        print(f"📊 {model_name} - RMSE: ₹{rmse:.2f}, R²: {r2:.4f}")
        
        return results
    
    def get_feature_importance(self, model, feature_names):
        """Get feature importance from ensemble model"""
        
        importances = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = dict(zip(feature_names, model.feature_importances_))
        elif hasattr(model, 'estimators_'):
            # For voting/stacking regressors
            avg_importance = np.zeros(len(feature_names))
            count = 0
            
            for name, estimator in model.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    avg_importance += estimator.feature_importances_
                    count += 1
            
            if count > 0:
                avg_importance /= count
                importances = dict(zip(feature_names, avg_importance))
        
        return importances
