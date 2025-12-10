#!/bin/bash

# Gone Phish'n — One-click installer & launcher (Mac & Linux)

echo "Hello and welcome to the Gone Phish'n Application"
echo

# 1. Install required packages
echo "Installing Python packages..."
pip3 install -r requirements.txt

# 2. Download NLTK data
echo "Downloading required language data..."
python3 << 'EOF'
import nltk, ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

for resource in ['stopwords', 'punkt', 'punkt_tab']:
    for resource in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"   → Downloading {resource}...")
            nltk.download(resource, quiet=True)
EOF

# 3. Train models if missing
if [ ! -f "phishing_model.pkl" ] || [ ! -f "email_spam_model.pkl" ]; then
    echo "Training AI models (first run only — ~60 seconds)..."
    python3 -m src.train_url
    python3 -m src.train_email
    echo "Models trained and saved!"
else
    echo "Models already exist — skipping training."
fi

# 4. Launch the app
echo
echo "Launching Gone Phish'n..."
echo "Close the window to quit."
echo

python3 main_gui.py