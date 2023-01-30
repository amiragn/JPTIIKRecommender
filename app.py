import pandas as pd
from flask import Flask, render_template, request
import pickle

import gensim.downloader as api
path = api.load("word2vec-google-news-300", return_path=True)
from gensim import models
wv = models.KeyedVectors.load_word2vec_format(r'C:\Users\Amira/gensim-data\word2vec-google-news-300\word2vec-google-news-300.gz', binary=True)

app = Flask(__name__)
model1 = pickle.load(open("model1.pkl", "rb"))
model2 = pickle.load(open("model2.pkl", "rb"))

def recommendation(title):
    matrix_article_title_vocab = []
    for list_ in model2[model2['title'].str.contains(title)].to_numpy():
        list_[2] = [word for word in list_[2] if word in wv]
        list_[4] = [word for word in list_[4] if word in wv]
        matrix_article_title_vocab.append(list_)

    matrix_similarity = []
    for list1 in model1.to_numpy():
        for list2 in matrix_article_title_vocab:
            try:
                score_abs = wv.n_similarity(list1[2], list2[2])
                score_title = wv.n_similarity(list1[4], list2[4])/2
            except ZeroDivisionError:
                score_title = 0      
            if ((list1[1] == list2[1]) & (list1[0] != list2[0])):
                matrix_similarity.append([list1[0], list1[3], score_title, score_abs])
    
    df_article_similarity = pd.DataFrame(matrix_similarity, columns = ['recommendation', 'link', 'score_title','score_abstract'])
    df_article_similarity['final_score'] = df_article_similarity['score_title'] + df_article_similarity['score_abstract']
    return (df_article_similarity.sort_values(by=['final_score', 'score_abstract', 'score_title'], ascending=False).head(10))

@app.route("/")
def home():
    try:
        return render_template("index.html")
    except Exception as e:
        return str(e)

@app.route("/recom", methods=["POST"])
def recom():
    string_feature = str(request.form.get("title"))
    recom = recommendation(string_feature)
    return render_template("index.html", tables=[recom.to_html(classes='data')], titles=['index','recommendation', 'link', 'score_title','score_abstract','final_score'])

if __name__ == "__main__":
    app.run(debug=True)