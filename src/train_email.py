import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess_email import preprocess
from model_email import MultinomialNB

def load_data():
    pd_CEAS_08 = pd.read_csv('../data/CEAS_08.csv')
    pd_Enron = pd.read_csv('../data/Enron.csv')
    pd_SpamAssasin = pd.read_csv('../data/SpamAssasin.csv')
    
    for df in [pd_CEAS_08, pd_Enron, pd_SpamAssasin]:
        df.drop(columns=['sender', 'receiver', 'date', 'urls', 'subject',], inplace=True, errors='ignore')
        
    mails = pd.concat([pd_CEAS_08, pd_Enron, pd_SpamAssasin])
    mails.rename(columns={'label': 'spam'}, inplace=True)
    mails['spam'] = mails['spam'].map({1: True, 0: False})
    mails.dropna(inplace=True)
    return mails

def train_model():
    data = load_data()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    model = MultinomialNB(preprocess_func=preprocess)
    model.fit(train_data)
    return model, test_data

if __name__ == "__main__":
    print("Training email spam detector...")
    model, test_data = train_model()
    
    model.save("../email_spam_model.pkl")
    
    print("Quick Test")
    print("Spam prob for 'win free money':", model.predict_proba("win free money click here"))
    print("Spam prob for 'meeting tomorrow':", model.predict_proba("meeting tomorrow at 3pm"))