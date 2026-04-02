import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text, keep_punctuation=True, lemmatize=False, remove_stopwords=False):
    """
    Enhanced text cleaning with configurable options
    """
    original_text = text

    text = text.lower()

    text = re.sub(r"^(dear|hi|hello|hey)\s+.*?,", "", text, flags=re.IGNORECASE)

    polite_phrases = [
        r"i hope this message finds you well",
        r"i hope this message reaches you",
        r"i hope you are doing well",
        r"i am reaching out to",
        r"just writing to",
        r"i wanted to",
        r"i would like to",
        r"i am writing to (report|inform|ask|request)",
        r"please (find|see) attached",
        r"attached please find"
    ]

    for phrase in polite_phrases:
        text = re.sub(phrase, "", text, flags=re.IGNORECASE)

    signoffs = [
        r"thank you.*$", r"thanks.*$", r"regards.*$",
        r"best regards.*$", r"sincerely.*$", r"cheers.*$",
        r"appreciate your (help|assistance|time).*$",
        r"looking forward.*$"
    ]

    for signoff in signoffs:
        text = re.sub(signoff, "", text, flags=re.IGNORECASE)

    text = re.sub(r"\b\w+@\w+\.\w+\b", "", text)
    text = re.sub(r"\b\d{10,}\b", "", text)  
    text = re.sub(r"ticket\s*id:?\s*\w+", "", text, flags=re.IGNORECASE) 
    if keep_punctuation:
        text = re.sub(r"[^\w\s\.\?\!]", "", text)
    else:
        text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords or lemmatize:
        tokens = text.split()

        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            domain_words = {'not', 'no', 'but', 'however', 'issue', 'problem', 'error'}
            stop_words = stop_words - domain_words
            tokens = [word for word in tokens if word not in stop_words]

        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        text = ' '.join(tokens)

    if not text or len(text.split()) < 2:
        text = original_text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

    return text