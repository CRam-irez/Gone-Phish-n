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
            n_estimators=400,
            random_state=42
        )
        
        self.p_spam = 0
        self.p_ham = 0
        
        # Text Model Train
        def fit_text_model(self, urls, labels):
            spam_urls = urls[labels == 1]
            ham_urls = urls[labels == 0]
            
            self.p_spam = len(spam_urls) / len(urls)
            self.p_ham = len(ham_urls) / len(urls)
            
            for url in spam_urls:
                words = preprocess_url(url)
                self.spam_word_counts.update(words)
                self.vocab.update(words)
                self.spam_total_words += len(words)
                
            for url in ham_urls:
                words = preprocess_url(url)
                self.ham_word_counts.update(words)
                self.vocab.update(words)
                self.ham_total_words += len(words)
            
        # Text Model Predict      
        def predict_text_proba(self, url):
            words = preprocess_url(url)
            vocab_size = len(self.vocab)
            
            log_spam = np.log(self.p_spam + 1e-9)
            log_ham = np.log(self.p_ham + 1e-9) 
            
            for w in words:
                log_spam += np.log((self.spam_word_counts[w] + 1) / (self.spam_total_words + vocab_size))
                log_ham += np.log((self.ham_word_counts[w] + 1) / (self.ham_total_words + vocab_size ))
                
            max_log = max(log_spam, log_ham)
            prob_spam = np.exp(log_spam - max_log)
            prob_ham = np.exp(log_ham - max_log)
            
            prob_spam /= (prob_spam + prob_ham)
            return prob_spam
        
        # Train Numeric Model
        def fit_numeric_model(self, X_numeric, labels):
            self.numeric_model.fit(X_numeric, labels)
            
        # Hybrid Prediction 
        def predict_proba(self, url, numeric_features):
            prob_text = self.predict_text_proba(url)
            prob_numeric = self.numeric_model.predict_proba([numeric_features])[0][1]
            
            return (prob_text + prob_numeric) / 2
        
        def predict(self, url, numeric_features):
            return self.predict_proba(url, numeric_features) > 0.5