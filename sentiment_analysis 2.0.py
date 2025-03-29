import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import requests
from textblob import TextBlob
import tweepy
from newsapi import NewsApiClient
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import json
import os

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self.news_api = NewsApiClient(api_key=config['news_api_key'])
        self.twitter_auth = tweepy.OAuthHandler(
            config['twitter_api_key'],
            config['twitter_api_secret']
        )
        self.twitter_auth.set_access_token(
            config['twitter_access_token'],
            config['twitter_access_token_secret']
        )
        self.twitter_api = tweepy.API(self.twitter_auth)
        
        # Initialize sentiment cache
        self.sentiment_cache = {}
        self.cache_duration = timedelta(hours=1)
        
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text using TextBlob
        """
        try:
            analysis = TextBlob(text)
            # Normalize sentiment score to [-1, 1]
            return analysis.sentiment.polarity
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return 0.0
    
    def get_news_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get sentiment from news articles
        """
        try:
            # Check cache first
            cache_key = f"news_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['sentiment']
            
            # Get news articles
            news = self.news_api.get_everything(
                q=symbol,
                language='en',
                from_param=(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                sort_by='relevancy'
            )
            
            if not news['articles']:
                return None
            
            # Analyze sentiment of each article
            sentiments = []
            for article in news['articles']:
                title_sentiment = self.analyze_text(article['title'])
                if article['description']:
                    desc_sentiment = self.analyze_text(article['description'])
                    sentiments.append((title_sentiment + desc_sentiment) / 2)
                else:
                    sentiments.append(title_sentiment)
            
            # Calculate weighted average sentiment
            avg_sentiment = np.mean(sentiments)
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'sentiment': avg_sentiment,
                'timestamp': datetime.now()
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Error getting news sentiment: {str(e)}")
            return None
    
    def get_twitter_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get sentiment from Twitter
        """
        try:
            # Check cache first
            cache_key = f"twitter_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['sentiment']
            
            # Get tweets
            tweets = self.twitter_api.search_tweets(
                q=f"${symbol}",
                lang="en",
                count=100
            )
            
            if not tweets:
                return None
            
            # Analyze sentiment of each tweet
            sentiments = []
            for tweet in tweets:
                sentiment = self.analyze_text(tweet.text)
                sentiments.append(sentiment)
            
            # Calculate weighted average sentiment
            avg_sentiment = np.mean(sentiments)
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'sentiment': avg_sentiment,
                'timestamp': datetime.now()
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {str(e)}")
            return None
    
    def get_market_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get market sentiment indicators
        """
        try:
            # Check cache first
            cache_key = f"market_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['sentiment']
            
            # Get market data
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Calculate various sentiment indicators
            sentiment_indicators = []
            
            # RSI sentiment
            if 'RSI' in info:
                rsi = info['RSI']
                rsi_sentiment = (rsi - 50) / 50  # Normalize to [-1, 1]
                sentiment_indicators.append(rsi_sentiment)
            
            # Volume sentiment
            if 'volume' in info and 'averageVolume' in info:
                volume_ratio = info['volume'] / info['averageVolume']
                volume_sentiment = (volume_ratio - 1) / volume_ratio  # Normalize to [-1, 1]
                sentiment_indicators.append(volume_sentiment)
            
            # Price momentum sentiment
            if 'regularMarketChangePercent' in info:
                momentum_sentiment = info['regularMarketChangePercent'] / 100
                sentiment_indicators.append(momentum_sentiment)
            
            if not sentiment_indicators:
                return None
            
            # Calculate weighted average sentiment
            avg_sentiment = np.mean(sentiment_indicators)
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'sentiment': avg_sentiment,
                'timestamp': datetime.now()
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return None
    
    def get_combined_sentiment(self, symbol: str) -> Optional[float]:
        """
        Get combined sentiment from all sources
        """
        try:
            # Check cache first
            cache_key = f"combined_{symbol}"
            if cache_key in self.sentiment_cache:
                cached_data = self.sentiment_cache[cache_key]
                if datetime.now() - cached_data['timestamp'] < self.cache_duration:
                    return cached_data['sentiment']
            
            # Get sentiment from all sources
            sentiments = []
            weights = []
            
            # News sentiment
            news_sentiment = self.get_news_sentiment(symbol)
            if news_sentiment is not None:
                sentiments.append(news_sentiment)
                weights.append(self.config['sentiment_weights']['news'])
            
            # Twitter sentiment
            twitter_sentiment = self.get_twitter_sentiment(symbol)
            if twitter_sentiment is not None:
                sentiments.append(twitter_sentiment)
                weights.append(self.config['sentiment_weights']['twitter'])
            
            # Market sentiment
            market_sentiment = self.get_market_sentiment(symbol)
            if market_sentiment is not None:
                sentiments.append(market_sentiment)
                weights.append(self.config['sentiment_weights']['market'])
            
            if not sentiments:
                return None
            
            # Calculate weighted average sentiment
            avg_sentiment = np.average(sentiments, weights=weights)
            
            # Cache the result
            self.sentiment_cache[cache_key] = {
                'sentiment': avg_sentiment,
                'timestamp': datetime.now()
            }
            
            return avg_sentiment
            
        except Exception as e:
            logger.error(f"Error getting combined sentiment: {str(e)}")
            return None
    
    def clear_cache(self):
        """
        Clear the sentiment cache
        """
        self.sentiment_cache.clear()
    
    def get_sentiment_signal(self, symbol: str) -> int:
        """
        Convert sentiment to trading signal
        """
        sentiment = self.get_combined_sentiment(symbol)
        if sentiment is None:
            return 0
        
        # Define sentiment thresholds
        positive_threshold = self.config['sentiment_thresholds']['positive']
        negative_threshold = self.config['sentiment_thresholds']['negative']
        
        if sentiment > positive_threshold:
            return 1  # Buy signal
        elif sentiment < negative_threshold:
            return -1  # Sell signal
        else:
            return 0  # Neutral signal 