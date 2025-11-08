import numpy as np
from collections import Counter 

class MultinomialNB:
    def __init__(self, preprocess_func):
        self.process = preprocess_func
        self.spam_word_counts = Counter()
        self.ham_word_counts = Counter()
        self.spam_total_words = 0
        self.ham_total_words = 0
        self.vocab = set()
        self.p_spam = 0
        self.p_ham = 0
        
    def fit(self, data):
        spam_mails = data[data['spam'] == True]
        ham_mails = data[data['spam'] == False]
        self.p_spam = len(spam_mails) / len(data)
        self.p_ham = len(ham_mails) / len(data)
        
        for text in spam_mails['body']:
            words = self.preprocess(text)
            self.spam_word_counts.update(words)
            self.vocab.update(words)
            self.spam_total_words += len(words)
            
        for text in ham_mails['body']:
            words = self.preprocess(text)
            self.ham_word_counts.update(words)
            self.vocab.update(words)
            self.ham_total_words += len(words)
            
    def predict_proba(self, text):
        words = self.preprocess(text)
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
        return prob_spam / norm
    
    def predict(self, text):
        return self.predict_proba(text) > 0.5