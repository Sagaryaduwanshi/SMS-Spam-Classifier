import os
import warnings
import nexmo
from flask import Flask, render_template, url_for, request, session
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# from dotenv import load_dotenv
# load_dotenv()

# API_KEY = os.getenv("API_KEY")
# API_SECRET = os.getenv("API_SECRET")
#  client = nexmo.Client(key=API_KEY, secret=API_SECRET)

warnings.filterwarnings("ignore")

app = Flask(__name__)
# secret key is needed for session
# app.secret_key = os.getenv('SECRET_KEY')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/inbox', methods=['POST'])
def inbox():
    """ A POST endpoint that sends an SMS. """

    # Extract the form values:
    to_number = request.form['to_number']
    message = request.form['message']
    # session['to_number'] = to_number
    # session['message'] = message

    # Send the SMS message:
    # result = client.send_message({
    #     'from': 'Vonage APIs',
    #     'to': to_number,
    #     'text': message,
    #  })
    return render_template('inbox.html', number=to_number, msg=message)

@app.route('/', methods=['POST'])
def predict():
    model = pickle.load(open("web_app/models/Naives_Bayes_Model.pkl", "rb"))
    tfidf_model = pickle.load(open("web_app/models/tfidf.pkl", "rb"))
    if request.method == "POST":
        message = str(session.get('message'))
        message = [message]
        dataset = {'message': message}
        data = pd.DataFrame(dataset)
        data["message"] = data["message"].str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddress')
        data["message"] = data["message"].str.replace(
            r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$', 'webaddress')
        data["message"] = data["message"].str.replace(r'£|\$', 'money-symbol')
        data["message"] = data["message"].str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3?[\d]{4}$', 'phone-number')
        data["message"] = data["message"].str.replace(r'\d+(\.\d+)?', 'number')
        data["message"] = data["message"].str.replace(r'[^\w\d\s]', ' ')
        data["message"] = data["message"].str.replace(r'\s+', ' ')
        data["message"] = data["message"].str.replace(r'^\s+|\s*?$', ' ')
        data["message"] = data["message"].str.lower()

        stop_words = set(stopwords.words('english'))
        data["message"] = data["message"].apply(lambda x: ' '.join(
            term for term in x.split() if term not in stop_words))
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import wordnet
        lemma = WordNetLemmatizer()
        data['message'] = data['message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.NOUN) for term in x.split()))
        data['message'] = data['message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.VERB) for term in x.split()))
        data['message'] = data['message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.ADJ) for term in x.split()))
        data['message'] = data['message'].apply(lambda x: ' '.join(lemma.lemmatize(term, wordnet.ADV) for term in x.split()))

        # tfidf_model = TfidfVectorizer()
        tfidf_vec = tfidf_model.transform(data["message"])
        tfidf_data = pd.DataFrame(tfidf_vec.toarray())
        my_prediction = model.predict(tfidf_data)

    return render_template('predict.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)