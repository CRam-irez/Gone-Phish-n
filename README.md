# Gone Phish'n - Phishing and Spam Detector Application

The purpose of this project was to create and train a naive Bayes classifer to determine if an email is spam or not.
Gone Phish'n is a desktop application that uses custom trained multinomial naive Bayes classifiers to detect malicious phishing URLS and spam emails in real time.

### Project Goal
Design, train, and implement two independent naive Bayes models:
- One trained on URL structure and domain patterns to detect phishing links
- One trained on email bodies to identify spam and scam content

### Key Features 
- Dual mode architecture (URL + Email classification)
- Custom feature extraction via tokenization and TD-IDF style probability scoring 
- Real time prediction 
- Multiline email input with full text analysis 
- Automatic model training and persistence using 'joblib'
