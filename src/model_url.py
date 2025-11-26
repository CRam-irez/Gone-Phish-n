import numpy as np
import joblib
from collections import Counter
from .preprocess_url import preprocess_url

# Text based phising detector that uses a custom Naive Bayes Classifier
class URLPhishingDetector:
    def __init__(self):
        
        # Text Based Naive Bayer
        self.spam_words = Counter()
        self.ham_words = Counter()
        self.spam_total = 0
        self.ham_total = 0
        self.vocab = set()
        
        # Prior Probabilities 
        self.p_spam = 0
        self.p_ham = 0
        
    # Train Model
    def fit(self, urls, labels):
        spam_urls = [u for u, l in zip(urls, labels) if l == 1]
        ham_urls = [u for u, l in zip(urls, labels) if l ==0]
            
        total = len(urls)
        self.p_spam = len(spam_urls) / total
        self.p_ham = len(ham_urls) / total
            
        # Count words in phising URLs
        for url in spam_urls:
            tokens = preprocess_url(url)
            self.spam_words.update(tokens)
            self.vocab.update(tokens)
            self.spam_total += len(tokens)
                
        # Count words in legit URLs 
        for url in ham_urls:
            tokens = preprocess_url(url)
            self.ham_words.update(tokens)
            self.vocab.update(tokens)
            self.ham_total += len(tokens)
            
    # Text Model Predict      
    def predict_proba(self, url):
        tokens = preprocess_url(url)
        if not tokens:
            return 0.5
        
        vocab_size = len(self.vocab) or 1
            
        log_spam = np.log(self.p_spam + 1e-9)
        log_ham = np.log(self.p_ham + 1e-9) 
            
        for token in tokens:
            spam_count = self.spam_words.get(token, 0)
            ham_count = self.ham_words.get(token, 0)
            log_spam += np.log((spam_count + 1) / (self.spam_total + vocab_size))
            log_ham += np.log((ham_count + 1) / (self.ham_total + vocab_size ))
                
        max_log = max(log_spam, log_ham)
        prob_spam = np.exp(log_spam - max_log)
        prob_ham = np.exp(log_ham - max_log) 
        return prob_spam / (prob_spam + prob_ham)
        
    def predict(self, url):
        return 1 if self.predict_proba(url) > 0.5 else 0
        
    # Save and Load 
    def save(self, path="phishing_model.pkl"):
        joblib.dump(self,path)
            
    @staticmethod
    def load(path="phishin_model.pkl"):
        return joblib.load(path)