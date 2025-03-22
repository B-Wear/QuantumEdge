import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gym
import logging
from typing import Dict, List, Tuple, Optional
import joblib
import os

logger = logging.getLogger(__name__)

class MachineLearningModel:
    def __init__(self, config: Dict):
        self.config = config
        self.model_type = config['model_type']
        self.features = config['features']
        self.scaler = StandardScaler()
        self.model = None
        self.env = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for machine learning
        """
        # Select features
        X = df[self.features].copy()
        
        # Create target variable (price movement direction)
        y = np.where(df['close'].shift(-1) > df['close'], 1, 0)
        
        # Drop NaN values
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_lstm(self, X: np.ndarray, y: np.ndarray):
        """
        Train LSTM model
        """
        # Reshape data for LSTM [samples, time steps, features]
        X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_reshaped, y,
            test_size=self.config['train_test_split'],
            random_state=42
        )
        
        # Create model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, X.shape[1]), return_sequences=True),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=self.config['validation_split'],
            verbose=0
        )
        
        # Evaluate model
        _, accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"LSTM model accuracy: {accuracy:.4f}")
        
        self.model = model
        return model
    
    def train_reinforcement_learning(self, df: pd.DataFrame):
        """
        Train reinforcement learning model
        """
        # Create custom trading environment
        self.env = TradingEnvironment(df)
        self.env = DummyVecEnv([lambda: self.env])
        
        # Create and train PPO model
        model = PPO(
            "MlpPolicy",
            self.env,
            learning_rate=self.config['rl']['learning_rate'],
            n_steps=self.config['rl']['n_steps'],
            batch_size=self.config['rl']['batch_size'],
            n_epochs=self.config['rl']['n_epochs'],
            gamma=self.config['rl']['gamma'],
            gae_lambda=self.config['rl']['gae_lambda'],
            clip_range=self.config['rl']['clip_range'],
            ent_coef=self.config['rl']['ent_coef'],
            vf_coef=self.config['rl']['vf_coef'],
            verbose=1
        )
        
        # Train model
        model.learn(total_timesteps=10000)
        
        self.model = model
        return model
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        if self.model_type == 'lstm':
            X_reshaped = np.reshape(X, (X.shape[0], 1, X.shape[1]))
            predictions = self.model.predict(X_reshaped)
        else:  # reinforcement learning
            predictions = self.model.predict(X)
        
        return predictions
    
    def save_model(self, filepath: str):
        """
        Save the trained model
        """
        if self.model is None:
            logger.error("Model not trained. Call train() first.")
            return None
        
        try:
            if self.model_type == 'lstm':
                self.model.save(filepath)
            else:
                self.model.save(f"{filepath}_rl")
            
            # Save scaler separately
            joblib.dump(self.scaler, f"{filepath}_scaler")
            
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str):
        """
        Load a trained model
        """
        try:
            if self.model_type == 'lstm':
                self.model = tf.keras.models.load_model(filepath)
            else:
                self.model = PPO.load(f"{filepath}_rl")
            
            # Load scaler
            self.scaler = joblib.load(f"{filepath}_scaler")
            
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning
    """
    def __init__(self, df: pd.DataFrame):
        super(TradingEnvironment, self).__init__()
        
        self.df = df
        self.current_step = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # Buy, Sell, Hold
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(df.columns),),
            dtype=np.float32
        )
    
    def reset(self):
        """
        Reset the environment
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0
        self.position_size = 0
        
        return self._get_observation()
    
    def step(self, action):
        """
        Execute one step in the environment
        """
        # Get current price
        current_price = self.df['close'].iloc[self.current_step]
        
        # Execute action
        reward = 0
        done = False
        
        if action == 0:  # Buy
            if self.position <= 0:
                self.position = 1
                self.position_size = self.balance / current_price
        elif action == 1:  # Sell
            if self.position >= 0:
                self.position = -1
                self.position_size = self.balance / current_price
        
        # Calculate reward
        next_price = self.df['close'].iloc[self.current_step + 1]
        price_change = (next_price - current_price) / current_price
        
        if self.position != 0:
            reward = self.position * price_change * 100
        
        # Update balance
        self.balance *= (1 + reward/100)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.df) - 1:
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """
        Get current observation
        """
        return self.df.iloc[self.current_step].values.astype(np.float32)
    
    def render(self):
        """
        Render the environment
        """
        pass
    
    def close(self):
        """
        Clean up resources
        """
        pass 