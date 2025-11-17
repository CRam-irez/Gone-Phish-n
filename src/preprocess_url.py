import re

def preprocess_url(url):
    
    url = url.lower()
    url = re.sub(r"https?://", "", url)
    url = re.sub(r"www\.", "", url)
    url = re.sub(r"[^a-z0-9:/?=&%._\-]", "" ,url)
    tokens = re.split(r"[./?=&-_]+", url)
    tokens = [t for t in tokens if len(t) > 2]
    return tokens