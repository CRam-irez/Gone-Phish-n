from urllib.parse import urlparse
import re
def preprocess_url(url):
    
    url = url.lower()
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path
    query = parsed.query
    
    # Split subdomains
    domain_tokens = re.split(r"[.]", domain)
    
    # Tokenize path + query 
    path_tokens = re.split(r"[\/\-_?=&]", path + " " + query)
    
    # Remove empty
    tokens = [t for t in domain_tokens + path_tokens if len(t) > 2]

    return tokens