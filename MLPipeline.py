# import libraries
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine
import pandas as pd
from itertools import chain

# load data from database
engine = create_engine('sqlite:///Anand_ETL_Pipeline_16022020.db')
df = pd.read_sql_table('Cyclone',engine)
df.head()


def tokenize(text):
    
    # Normalize text and Tokenize text
    words = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    #print(words)

    # Reduce words to their root form
    lemmed_before = [WordNetLemmatizer().lemmatize(w) for w in words]
    #print(lemmed_before)

    # Lemmatize verbs by specifying pos
    lemmed_after = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed_before]
    #print(lemmed_after)
    
    return lemmed_after


token = lambda x: tokenize(x)


X = [token(df.message[x]) for x in range(len(df['message']))]
#Y =
X = list(chain(*X))

X


