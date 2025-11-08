import pandas as pd 
from sklearn.model_selection import train_test_split
from .model_url import MultinomialNB_URL

def train_url_model():
    data = pd.read_csv("data/URL_Dataset.csv")
    
    data.rename(columns={"ClassLabel": "label"}, inplace=True)
    data["label"] = data["label"].map({0: "spam", 1: "legit"})
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    model = MultinomialNB_URL()
    model.fit(train_data)
    
    correct = 0
    for _, row in test_data.iterrows():
        pred = model.predict(row['URL'])
        if (pred and row['label'] == 'legit') or (not pred and row['label'] == 'spam'):
            correct += 1
            
    accuracy = correct / len(test_data)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return model

if __name__ == "__main__":
    train_url_model()