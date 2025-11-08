import numpy as np
from collections import Counter
from .preprocess_url import preprocess_url

class MultinomialNB_URL:
    def __init__(self):
        self.spam_word_counts = Counter()
        self.ham_word_counts = Counter()
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.vocab = set()
        self.p_spam = 0
        self.p_ham = 0
        
    def fit(self, data):
        spam_urls = data[data['label'] == 'spam']
        ham_urls = data[data['label'] == 'legit']
        
        self.p_spam = len(spam_urls) / len(data)
        self.p_ham = len(ham_urls) / len(data)
        
        for url in spam_urls['URL']:
            words = preprocess_url(url)
            self.spam_word_counts.update(words)
            self.vocab.update(words)
            self.spam_total_words += len(words)
            
        for url in ham_urls['URL']:
            words = preprocess_url(url)
            self.ham_word_counts.update(words)
            self.vocab.update(words)
            self.ham_total_words += len(words)
            
    def predict_proba(self, url):
        words = preprocess_url(url)
        vocab_size = len(self.vocab)
        
        log_spam = np.log(self.p_spam)
        log_ham = np.log(self.p_ham)
        
        for w in words:
            log_spam += np.log((self.spam_word_counts[w] + 1) / (self.spam_total_words + vocab_size))
            log_ham += np.log((self.ham_word_counts[w] + 1) / (self.ham_total_words + vocab_size))
            
        max_log = max(log_spam, log_ham)
        prob_spam = np.exp(log_spam - max_log)
        prob_ham = np.exp(log_ham - max_log)
        norm = prob_spam + prob_ham
        prob_spam /= norm
        prob_ham /= norm
        return prob_spam
    
    def predict(self, url):
        return self.predict_proba(url) > 0.5