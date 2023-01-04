import pickle
import string
import streamlit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl','rb'))

model = pickle.load(open('model.pkl','rb'))



# 1. preprocessing



def transform_text(text):
    text = text.lower() #convert to lower case
    text = nltk.word_tokenize(text) #seprate words from sentence
    y = []
    for i in text:
        if i.isalnum(): #checking if word is only alphabet numeric
            y.append(i) 
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

msg = input("enter:")
transformed_sms = transform_text(msg)


# 2. vectrize

vector = tfidf.transform([transformed_sms])
# 3. predict

result = model.predict(vector)[0]
# 4. display

if result == 1:
    print("spam")
else:
    print("not spam")
