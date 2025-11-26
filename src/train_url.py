import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from .model_url import URLPhishingDetector

def train_and_save():
    print("Loading dataset...")
    data = pd.read_csv("data/URL_Dataset.csv", on_bad_lines='skip',engine='python')
    data = data[pd.to_numeric(data["label"], errors="coerce").notnull()]
    data["label"] = data["label"].astype(int)
    
    urls = data["domain"].astype(str)
    labels = data["label"]
    
    # Split and Train
    urls_train, urls_test, y_train, y_test = train_test_split(
        urls, labels,
        test_size=0.2,
        random_state=42
    )
    
    model = URLPhishingDetector()
    model.fit(urls_train.values, y_train.values)
    
    # Test Accuracy
    preds = [model.predict(url) for url in urls_test]
    acc = accuracy_score(y_test, preds)
    print(f"Text-Only NB Model Accuracy: {acc:.1%}")
    
    # Save 
    model.save("phishing_model.pkl")
    print("Model Saved -> phishing_model.pkl")
    return model
    
if __name__ == "__main__":
    train_and_save()