from googletrans import Translator
import sys
import json
import time
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import Stream
import tweepy
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import streamlit as st
import pandas as pd

import pickle
st.title("Predict MBTI personality by Twitter", anchor=None)


def Cleaning(text):
    import re
    import string
    # case folding
    text = text.lower()
    text = text.strip()
    text = re.sub('\s+', ' ', text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    # r memberi tahu python bawah the expression is raw string
    text = re.sub(r'@[A-Za-z-0-9]+', '', text)
   # for removing hashtag
    text = re.sub(r'#', '', text)
    # remove rt
    text = re.sub(r'RT[\s]+', '', text)  # Removing RT
    text = re.sub(r'https?:\/\/\S+', '', text)  # removelink
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    regrex_pattern = re.compile(pattern="["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U0001F98B   "  # butterfly
                                u"\U00002764  "  # redlove
                                u"\U0001F9E1  "  # orangelove
                                u"\u270C"  # peace
                                u"\U0001f970"  # face berbunga-bunga
                                u"\U0001f92f"  # kepala meledak
                                u"\U0001f92c"  # angry
                                u"\U0001f92d"  # malu
                                u"\U0001f924"  # melted
                                u"\U0001f923"  # ngakak
                                u"\U0001f929"  # matalove
                                u"\U0001f921"  # Badut
                                u"\u2763\ufe0f"  # Love
                                u"\U0001f917"  # hug
                                u"\U0001f92a"  # matabesar sebelah
                                u"\u2615"  # kopi
                                u"\U0001f914"  # emot ga tau
                                u"\U0001f91d"  # jabat tangan
                                u"\U0001f97a"  # mata berkaca-kaca
                                u"\U0001f925"  # Hidung pinokio
                                u"\u2639\ufe0f"  # sad
                                u"\U0001f90d"  # loveputih
                                u"\u2728"  # gemerlap bintang
                                u"\U0001f911"  # mataduitan
                                u"\U0001f98d"  # Kingkong
                                u"\U0001f9c2"  # merica




                                "]+", flags=re.UNICODE)
    return regrex_pattern.sub(r' ', text)

    # Preprocces


def prepro(text):
    from nltk.tokenize import word_tokenize
    import nltk
    from nltk.corpus import stopwords

    # word Tokenize
    text = word_tokenize(text)

    # Stopword
    list_stopwords = stopwords.words('indonesian')
    list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo',
                           'kalo', 'amp', 'biar', 'bikin', 'bilang',
                           'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                           'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                           'jd', 'jgn', 'sdh', 'aja', 'n', 't',
                           'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                           '&amp', 'yah', 'bak', 'haii', 'wkwkwkk', ])
    txt_stopword = pd.read_csv(
        "Dataset/stopwords.txt", names=["stopwords"], header=None)
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)
    text = [word for word in text if word not in list_stopwords]

    # normalisasi kata
    normalizad_word = pd.read_excel("Dataset/Normalisasi.xlsx")
    normalizad_word_dict = {}
    for index, row in normalizad_word.iterrows():
        if row[0] not in normalizad_word_dict:
            normalizad_word_dict[row[0]] = row[1]
            text = [normalizad_word_dict[term]
                    if term in normalizad_word_dict else term for term in text]
            return text


def showtfidf(Xtfidf):
    # show example dtraining tfidf
    tfidf_df = pd.DataFrame(Xtfidf, index=list_Tweet,
                            columns=cntizer.get_feature_names())
    tfidf_df = tfidf_df.stack().reset_index()
    tfidf_df = tfidf_df.rename(
        columns={0: 'Xtfidf', 'level_0': 'document', 'level_1': 'term', 'level_2': 'term'})
    show = tfidf_df.sort_values(by=['document', f'{Xtfidf}'], ascending=[
                                True, False]).groupby(['document']).head()
    return show


# translate class into binary
b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'},
               {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector

    return [b_Pers[l] for l in personality]


def translate_back(personality):
    # transform binary vector to mbti personality

    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


datapr = pd.read_pickle("./pickle-data/post-after-prepro.pkl")
stc = [''.join(str(item)) for item in datapr.prepro]


# Posts to a matrix of token counts
cntizer = CountVectorizer(analyzer="word",
                          max_features=1500,
                          tokenizer=None,
                          preprocessor=None,
                          stop_words=None,
                          max_df=0.7,
                          min_df=0.1)

# Learn the vocabulary dictionary and return term-document matrix
print("CountVectorizer...")
X_cnt = cntizer.fit_transform(stc)

# Transform the count matrix to a normalized tf or tf-idf representation
tfizer = TfidfTransformer()

print("Tf-idf...")
# Learn the idf vector (fit) and transform a count matrix to a tf-idf representation
X_tfidf = tfizer.fit_transform(X_cnt).toarray()

print("Tf-idf success")


# X = X_tfidf
# Y = datapr.type


# test_size = 0.20
# X_train, X_test, Y_train, Y_test = train_test_split(
#     X, Y, test_size=test_size, random_state=0)


# # Creating the SVM model
# model = OneVsRestClassifier(SVC())

# # Fitting the model with training data
# model.fit(X_train, Y_train)

# # Making a prediction on the test set
# prediction = model.predict(X_test)

# print("modeling success..")

# Evaluating the model
# print(f"Test Set Accuracy : {accuracy_score(Y_test, prediction) * 100} %\n\n")

def crawling(uname):
    access_token = "3282754098-bquw69Xy3gnLLsa7ONBnmJTyho9xsU32J0cMu3K"
    access_token_secret = "vIWpHIDdSyAfX0VmhfkPQZZddWF0436Bn3e2OBt83nIZW"
    consumer_key = "bCN1RGObGNZErZVlCOzVDQzxY"
    consumer_secret = "uscpIUw6ikn4JfDdPlP5ZJUiZR8wKr6HTCOCj0Q8SqIVvkybg3"
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    try:

        posts = api.user_timeline(
            screen_name=uname, count=100, lang="id", tweet_mode="extended")

    except tweepy.TweepError:
        print("Failed to run the command on that user, Skipping...")
        sys.exit(1)

    # membuat dataframe dan sebuah collumn yang kita akan namakan tweet
    df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])

    df = df[~df['Tweets'].astype(str).str.startswith('RT')]

    df['MBTI'] = 'unknown'

    test = df.groupby('MBTI')['Tweets'].apply(' '.join).reset_index()
    testproxies = "http://myproxy.org"
    translator = Translator()

    test['Tweet_id'] = test['Tweets'].apply(translator.translate, dest='id')
    test['Tweet_id'] = test['Tweet_id'].apply(getattr, args=('text',))

    test['Tweet_id'] = test['Tweet_id'].apply(Cleaning)
    print("Cleaning success ..")
    test['Tweet_id'] = test['Tweet_id'].apply(prepro)
    print("Preprocessing success ..")
    Tweet = [''.join(str(item)) for item in test.Tweet_id]

    print(Tweet)
    my_X_cnt = cntizer.transform(Tweet)

    with open('pipe.pickle', 'rb') as picklefile:
        saved_pipe = pickle.load(picklefile)

    # saved_pipe.predict(X_test)

    # my_X_cnt = cntizer.transform(Tweet)
    my_X_tfidf = tfizer.transform(my_X_cnt).toarray()
    y_pr = saved_pipe.predict(my_X_tfidf)
    print("success ..")
    print(F"Prediction is : {y_pr}")
    user = api.get_user(screen_name=f"{uname}")
    st.subheader(F"twitter name : {user.name}")

    st.subheader(f"prediction is : {y_pr[0]}")
    # st.subheader(f"Deskripsi : ")
    if y_pr[0] == "INTJ":
        st.subheader(
            " Deskripsi : Pemikir yang imajinatif dan strategis, dengan rencana untuk segala sesuatunya.")
    elif y_pr[0] == "INTP":
        st.subheader(
            "Deskripsi : Belajar dari hari kemarin, hidup untuk hari ini, berharap untuk hari esok. Hal penting adalah tidak berhenti bertanya.")
    elif y_pr[0] == "ENTJ":
        st.subheader(
            "Deskripsi : Pemimpin yang pemberani, imaginatif dan berkemauan kuat, selalu menemukan cara - atau menciptakan cara.")
    elif y_pr[0] == "ENTP":
        st.subheader(
            "Deskripsi : Pemikir yang cerdas dan serius yang gatal terhadap tantangan intelektual.")
    elif y_pr[0] == "INFJ":
        st.subheader(
            "Deskripsi : Pendiam dan mistis, tetapi idealis yang sangat menginspirasi dan tak kenal lelah.")
    elif y_pr[0] == "INFP":
        st.subheader(
            "Deskripsi : Orang yang puitis, baik hati dan altruisik, selalu ingin membantu aksi kebaikan.")
    elif y_pr[0] == "ENFJ":
        st.subheader(
            "Deskripsi : Pemimpin yang karismatik dan menginspirasi, mampu memukai pendengarnya.")
    elif y_pr[0] == "ENFP":
        st.subheader(
            "Deskripsi : Semangat yang antusias, kreatif dan bebas bergaul, yang selalu dapat menemukan alasan untuk tersenyum.")
    elif y_pr[0] == "ISTJ":
        st.subheader(
            "Deskripsi : Individu yang praktis dan mengutamakan fakta, yang keandalannya tidak dapat diragukan.")
    elif y_pr[0] == "ISFJ":
        st.subheader(
            "Deskripsi : Pelindung yang sangat berdedikasi dan hangat, selalu siap membela orang yang dicintainya.")
    elif y_pr[0] == "ESTJ":
        st.subheader(
            "Deskripsi : Administrator istimewa, tidak ada duanya dalam mengelola sesuatu - atau orang.")
    elif y_pr[0] == "ESFJ":
        st.subheader(
            "Deskripsi : Orang yang sangat peduli, sosial dan populer, selalu ingin membantu.")
    elif y_pr[0] == "ISTP":
        st.subheader(
            "Deskripsi : Eksperimenter yang pemberani dan praktis, menguasai semua jenis alat.")
    elif y_pr[0] == "ISFP":
        st.subheader(
            "Deskripsi : Seniman yang fleksibel dan mengagumkan, selalu siap menjelajahi dan mengalami hal baru.")
    elif y_pr[0] == "ESTP":
        st.subheader(
            "Deskripsi : Orang yang cerdas, bersemangan dan sangat tanggap, yang benar-benar menikmati hidup yang menantang.")
    elif y_pr[0] == "ESFP":
        st.subheader(
            "Deskripsi : Orang yang spontan, bersemangan dan antusias - hidup tidak akan membosankan saat berdekatan dengan mereka.")


uname = st.text_input('Enter your username:', placeholder="@username")

crawling(uname)


# uname = input("Enter your username: ")
