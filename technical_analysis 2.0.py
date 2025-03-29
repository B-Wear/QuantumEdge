import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.indicators = {}
        self.patterns = {}
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the dataframe
        """
        df = df.copy()
        
        # Trend Indicators
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
        
        # Volatility Indicators
        df['upperband'], df['middleband'], df['lowerband'] = talib.BBANDS(
            df['close'],
            timeperiod=self.config['indicators']['bollinger_bands']['period'],
            nbdevup=self.config['indicators']['bollinger_bands']['std_dev'],
            nbdevdn=self.config['indicators']['bollinger_bands']['std_dev']
        )
        df['atr'] = talib.ATR(
            df['high'],
            df['low'],
            df['close'],
            timeperiod=self.config['indicators']['atr']['period']
        )
        
        # Momentum Indicators
        df['rsi'] = talib.RSI(
            df['close'],
            timeperiod=self.config['indicators']['rsi']['period']
        )
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(
            df['close'],
            fastperiod=self.config['indicators']['macd']['fast_period'],
            slowperiod=self.config['indicators']['macd']['slow_period'],
            signalperiod=self.config['indicators']['macd']['signal_period']
        )
        
        # Volume Indicators
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Additional Indicators
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        df['stoch_k'], df['stoch_d'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowd_period=3
        )
        
        # Store indicators for reference
        self.indicators = {
            'trend': ['sma_20', 'sma_50', 'sma_200', 'ema_20'],
            'volatility': ['upperband', 'middleband', 'lowerband', 'atr'],
            'momentum': ['rsi', 'macd', 'macdsignal', 'macdhist'],
            'volume': ['obv'],
            'additional': ['adx', 'cci', 'stoch_k', 'stoch_d']
        }
        
        return df
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect candlestick patterns and chart patterns
        """
        df = df.copy()
        
        # Candlestick Patterns
        if self.config['patterns']['candlestick']:
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Chart Patterns
        if self.config['patterns']['chart']:
            df['head_and_shoulders'] = self._detect_head_and_shoulders(df)
            df['double_top'] = self._detect_double_top(df)
            df['double_bottom'] = self._detect_double_bottom(df)
            df['triangle'] = self._detect_triangle(df)
        
        return df
    
    def _detect_head_and_shoulders(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect head and shoulders pattern
        """
        pattern = pd.Series(0, index=df.index)
        
        for i in range(20, len(df) - 20):
            # Find potential left shoulder
            left_shoulder = df['high'].iloc[i-20:i].max()
            left_shoulder_idx = df['high'].iloc[i-20:i].idxmax()
            
            # Find potential head
            head = df['high'].iloc[i-10:i+10].max()
            head_idx = df['high'].iloc[i-10:i+10].idxmax()
            
            # Find potential right shoulder
            right_shoulder = df['high'].iloc[i:i+20].max()
            right_shoulder_idx = df['high'].iloc[i:i+20].idxmax()
            
            # Find neckline
            neckline = min(df['low'].iloc[left_shoulder_idx:right_shoulder_idx])
            
            # Check pattern conditions
            if (left_shoulder < head and right_shoulder < head and
                abs(left_shoulder - right_shoulder) / head < 0.1):
                pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_double_top(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect double top pattern
        """
        pattern = pd.Series(0, index=df.index)
        
        for i in range(20, len(df) - 20):
            # Find potential first peak
            first_peak = df['high'].iloc[i-20:i].max()
            first_peak_idx = df['high'].iloc[i-20:i].idxmax()
            
            # Find potential second peak
            second_peak = df['high'].iloc[i:i+20].max()
            second_peak_idx = df['high'].iloc[i:i+20].idxmax()
            
            # Find valley between peaks
            valley = df['low'].iloc[first_peak_idx:second_peak_idx].min()
            
            # Check pattern conditions
            if (abs(first_peak - second_peak) / first_peak < 0.02 and
                (first_peak - valley) / first_peak > 0.02):
                pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_double_bottom(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect double bottom pattern
        """
        pattern = pd.Series(0, index=df.index)
        
        for i in range(20, len(df) - 20):
            # Find potential first bottom
            first_bottom = df['low'].iloc[i-20:i].min()
            first_bottom_idx = df['low'].iloc[i-20:i].idxmin()
            
            # Find potential second bottom
            second_bottom = df['low'].iloc[i:i+20].min()
            second_bottom_idx = df['low'].iloc[i:i+20].idxmin()
            
            # Find peak between bottoms
            peak = df['high'].iloc[first_bottom_idx:second_bottom_idx].max()
            
            # Check pattern conditions
            if (abs(first_bottom - second_bottom) / first_bottom < 0.02 and
                (peak - first_bottom) / first_bottom > 0.02):
                pattern.iloc[i] = 1
        
        return pattern
    
    def _detect_triangle(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect triangle patterns (ascending, descending, symmetrical)
        """
        pattern = pd.Series(0, index=df.index)
        
        for i in range(20, len(df) - 20):
            # Get highs and lows for the period
            highs = df['high'].iloc[i-20:i]
            lows = df['low'].iloc[i-20:i]
            
            # Calculate trend lines
            high_slope = np.polyfit(range(len(highs)), highs, 1)[0]
            low_slope = np.polyfit(range(len(lows)), lows, 1)[0]
            
            # Classify triangle type
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                pattern.iloc[i] = 1  # Ascending triangle
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                pattern.iloc[i] = 2  # Descending triangle
            elif abs(high_slope + low_slope) < 0.001:
                pattern.iloc[i] = 3  # Symmetrical triangle
        
        return pattern
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators and patterns
        """
        df = df.copy()
        
        # Initialize signal column
        df['signal'] = 0
        
        # RSI signals
        df.loc[df['rsi'] < self.config['indicators']['rsi']['oversold'], 'signal'] += 1
        df.loc[df['rsi'] > self.config['indicators']['rsi']['overbought'], 'signal'] -= 1
        
        # MACD signals
        df.loc[df['macd'] > df['macdsignal'], 'signal'] += 1
        df.loc[df['macd'] < df['macdsignal'], 'signal'] -= 1
        
        # Bollinger Bands signals
        df.loc[df['close'] < df['lowerband'], 'signal'] += 1
        df.loc[df['close'] > df['upperband'], 'signal'] -= 1
        
        # Trend signals
        df.loc[df['close'] > df['sma_20'], 'signal'] += 1
        df.loc[df['close'] < df['sma_20'], 'signal'] -= 1
        
        # Pattern signals
        if 'head_and_shoulders' in df.columns:
            df.loc[df['head_and_shoulders'] == 1, 'signal'] -= 1
        if 'double_top' in df.columns:
            df.loc[df['double_top'] == 1, 'signal'] -= 1
        if 'double_bottom' in df.columns:
            df.loc[df['double_bottom'] == 1, 'signal'] += 1
        if 'triangle' in df.columns:
            df.loc[df['triangle'] == 1, 'signal'] += 1  # Ascending
            df.loc[df['triangle'] == 2, 'signal'] -= 1  # Descending
        
        # Normalize signals to -1, 0, 1
        df['signal'] = df['signal'].apply(lambda x: 1 if x > 2 else (-1 if x < -2 else 0))
        
        return df
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate support and resistance levels
        """
        support = df['low'].rolling(window=window).min()
        resistance = df['high'].rolling(window=window).max()
        
        return support, resistance
    
    def calculate_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate various volatility measures
        """
        # ATR-based volatility
        atr_volatility = df['atr'] / df['close']
        
        # Bollinger Band width
        bb_width = (df['upperband'] - df['lowerband']) / df['middleband']
        
        # Historical volatility
        returns = df['close'].pct_change()
        hist_volatility = returns.rolling(window=20).std()
        
        return pd.DataFrame({
            'atr_volatility': atr_volatility,
            'bb_width': bb_width,
            'hist_volatility': hist_volatility
        })
    
    def get_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine market regime (trending, ranging, volatile)
        """
        regime = pd.Series('unknown', index=df.index)
        
        # Calculate ADX for trend strength
        adx = df['adx']
        
        # Calculate volatility
        volatility = self.calculate_volatility(df)['atr_volatility']
        
        # Determine regime
        regime.loc[adx > 25] = 'trending'
        regime.loc[(adx <= 25) & (volatility > volatility.rolling(window=20).mean())] = 'volatile'
        regime.loc[(adx <= 25) & (volatility <= volatility.rolling(window=20).mean())] = 'ranging'
        
        return regime 