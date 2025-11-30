from abc import ABC, abstractmethod
from typing import Dict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class SentimentAnalyzer(ABC):
    """
    Abstract base class for sentiment analysis.
    """
    @abstractmethod
    def analyze(self, text: str) -> Dict[str, float]:
        """
        Analyze the sentiment of a given text.
        Returns a dictionary with 'neg', 'neu', 'pos', 'compound' scores.
        """
        pass

class VaderSentimentAnalyzer(SentimentAnalyzer):
    """
    Sentiment analyzer using VADER.
    """
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict[str, float]:
        try:
            return self.analyzer.polarity_scores(str(text))
        except Exception:
            return {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}
