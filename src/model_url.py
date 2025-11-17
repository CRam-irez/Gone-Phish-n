import numpy as np
from collections import Counter
from .preprocess_url import preprocess_url
from sklearn.ensemble import RandomForestClassifier

class HybridURLClassifier:
    def __init__(self):
        
        # Text Model
        self.spam_word_counts = Counter()
        self.ham_word_counts = Counter()
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.vocab = set()
        
        # Numeric Model 
        self.numeric_model = RandomForestClassifier(
            n_estimators=400
            random_state=42
        )
        
        self.p_spam = 0
        self.p_ham = 0