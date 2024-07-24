
# coding: utf-8

# # SMS SPAM Classifier...

# In[1]:





# importing required libraries and functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import wordcloud
import nltk
import warnings
warnings.filterwarnings('ignore')


# Data Extraction and Visualisation

# In[2]:


# reading csv
file = pd.read_csv("spam.csv",encoding="latin1")
file.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
file.rename(columns = {'v1':'Label','v2':'Message'},inplace=True)



# In[3]:


fig = px.histogram(file, x="Label", color="Label", color_discrete_sequence=["#871fff","#ffa78c"])
fig.show()


# In[4]:


file['Length'] = file['Message'].apply(len)



# In[5]:


fig = px.histogram(file[file["Label"]=="ham"], x="Length", color="Label", color_discrete_sequence=["#871fff"] )
fig.show()
fig = px.histogram(file[file["Label"]=="spam"], x="Length", color="Label", color_discrete_sequence=["#ffa78c"] )
fig.show()


# In[6]:


data_ham  = file[file['Label'] == "ham"].copy()
data_spam = file[file['Label'] == "spam"].copy()

def white_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("white")

def show_wordcloud(file, title):
    text = ' '.join(file['Message'].astype(str).tolist())
    stopwords = set(wordcloud.STOPWORDS)
    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords, background_color="black",
                                        width = 3000, height = 2000).generate(text)
    # set the word color to black
    fig_wordcloud.recolor(color_func = white_color_func)
    plt.figure(figsize=(15,15), frameon=True)
    plt.imshow(fig_wordcloud)  
    plt.axis('off')
    plt.title(title, fontsize=20)
    plt.show()


# In[7]:


show_wordcloud(data_spam, "Spam messages")


# In[8]:


show_wordcloud(data_ham, "ham messages")


# In[9]:


file['Label'] = file['Label'].map( {'spam': 1, 'ham': 0})




# In[10]:


# Replace email address with 'emailaddress'
file['Message'] = file['Message'].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')

# Replace urls with 'webaddress'
file['Message'] = file['Message'].str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')

# Replace money symbol with 'money-symbol'
file['Message'] = file['Message'].str.replace(r'£|\$', 'money-symbol')

# Replace 10 digit phone number with 'phone-number'
file['Message'] = file['Message'].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3?[\d]{4}$', 'phone-number')

# Replace normal number with 'number'
file['Message'] = file['Message'].str.replace(r'\d+(\.\d+)?', 'number')

# remove punctuation
file['Message'] = file['Message'].str.replace(r'[^\w\d\s]', ' ')

# remove whitespace between terms with single space
file['Message'] = file['Message'].str.replace(r'\s+', ' ')

# remove leading and trailing whitespace
file['Message'] = file['Message'].str.replace(r'^\s+|\s*?$', ' ')

# change words to lower case
file['Message'] = file['Message'].str.lower()



# In[11]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
file['Message'] = file['Message'].apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))



# In[12]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
lemma = WordNetLemmatizer()
file['Message'] = file['Message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.NOUN) for term in x.split()))
file['Message'] = file['Message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.VERB) for term in x.split()))
file['Message'] = file['Message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.ADJ) for term in x.split()))
file['Message'] = file['Message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.ADV) for term in x.split()))
# print(lemma.lemmatize('dogs', wordnet.NOUN ))
# print(lemma.lemmatize('going', pos = 'v' ))
# print(lemma.lemmatize('better', wordnet.ADJ ))
# print(lemma.lemmatize('quickly', wordnet.ADV ))



# In[20]:


from nltk.tokenize import word_tokenize
sms_df = file['Message']
# creating a bag-of-words model
all_words = []
for sms in sms_df:
    words = word_tokenize(sms)
    for w in words:
        all_words.append(w)
all_words = nltk.FreqDist(all_words)


# In[27]:


# all_words.plot()
all_words.plot(15, title='Top 10 Most Common Words')


# In[ ]:
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer()
tfidf_vec=tfidf_model.fit_transform(sms_df)
import pickle
# serializing our model to a file called model.pkl
pickle.dump(tfidf_model, open("tfidf_model.pkl","wb"))
tfidf_data=pd.DataFrame(tfidf_vec.toarray())
tfidf_data.head()





# %%
target = file['Label']
tfidf_data['Label'] =  target
x_label =  tfidf_data['Label']
x = tfidf_data.drop('Label',axis = 1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,x_label,random_state=42,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)
  
# making predictions on the testing set
y_pred = gnb.predict(x_test)
  
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# %%
tfidf_data.drop("Label",inplace=True,axis = 1)
gnb.fit(tfidf_data,target)

pickle.dump(gnb,open('Naives_Bayes_Model.pkl','wb'))