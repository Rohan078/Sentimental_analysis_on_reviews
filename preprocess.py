import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def clean(text):
    text = str(text).lower()
    
    
    text = text.replace("read more", "")
    
    text = re.sub(r'[^a-z ]', '', text)

    sw = stopwords.words('english')
   
    negations = {'no', 'not', 'nor', 'never', 'none', 'neither', 'scarcely', 'barely', 'doesn', 'isn', 'wasn', 'shouldn', 'wouldn', 'couldn', 'won', 't', 'don'}
  
    
    sw = [word for word in sw if word not in negations]
    
    
    sw.extend(['read', 'more']) 
    
    text = " ".join([w for w in text.split() if w not in sw])

    lem = WordNetLemmatizer()
    text = " ".join([lem.lemmatize(w) for w in text.split()])

    return text
