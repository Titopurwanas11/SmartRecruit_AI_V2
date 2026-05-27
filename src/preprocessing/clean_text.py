import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def clean_text(text: str) -> str:
    """
    Input:  raw string (resume text or job description)
    Output: cleaned lowercase string — no special chars, no stopwords, no extra whitespace
    Steps:  lowercase → remove URLs → remove special chars → tokenize → remove stopwords → rejoin
    """
    if text is None or not isinstance(text, str) or text.strip() == "":
        return ""
    
    # Download NLTK data (once with guard)
    # Note: nltk.download is smart enough to skip if already present, 
    # but we follow the instruction to keep it inside with quiet=True.
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    
    # Remove special chars (preserve spaces and alphanumeric)
    # string.punctuation: '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english')).union(set(stopwords.words('indonesian')))
    filtered_tokens = [w for w in tokens if w not in stop_words]
    
    # Rejoin
    return " ".join(filtered_tokens)
