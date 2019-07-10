#Import Flask Library
from flask import Flask, render_template, request, session, url_for, redirect
import pandas as pd
import json
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from autocorrect import spell
from operator import itemgetter
import operator
#Initialize the app from Flask
app = Flask(__name__)


with app.open_resource('data/data.json') as f:
	master_df = pd.DataFrame.from_records(json.load(f))
with open('data/data.json') as f:
    data = json.load(f)
myvectorizer = pickle.load(open("data/vectorizer.pickle", "rb"))
tfidf = pickle.load(open("data/tfidf.pickle", "rb"))

@app.route('/search/<string:query>', methods=('GET', 'POST'))
def search(query):
        if request.method == 'POST':
                query = request.form['search_field']
                query=query
                return redirect(url_for('search', query=query))

        query_string = [query]
        re_item = query 
        query_vec = myvectorizer.transform(query_string)
        cosine_similarities = linear_kernel(query_vec, tfidf).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-21:-1]
        filtered_docs_indices = [i for i in related_docs_indices if (cosine_similarities[i] > 0)]

        #regular expression
        for i,each in enumerate(data): 
            if re.search(re_item, each['name'], flags=re.IGNORECASE):
                    filtered_docs_indices.append(i)
            for word in each['name'].split(' '):
                for look in query.split(' '):
                    if re.search(look, word, flags=re.IGNORECASE):
                        filtered_docs_indices.append(i)
                        
        for i,each in enumerate(data): 
            if re.search(re_item, each['title'], flags=re.IGNORECASE):
                    filtered_docs_indices.append(i)
            for word in each['title'].split(' '):
                for look in query.split(' '):
                    if re.search(look, word, flags=re.IGNORECASE):
                        filtered_docs_indices.append(i)                
        filtered_docs_indices = list(dict.fromkeys(filtered_docs_indices))

        if len(filtered_docs_indices) == 0:
            query_string = [spell(query)]
            re_item = spell(query)
            query_vec = myvectorizer.transform(query_string)
            cosine_similarities = linear_kernel(query_vec, tfidf).flatten()
            related_docs_indices = cosine_similarities.argsort()[:-21:-1]
            filtered_docs_indices = [i for i in related_docs_indices if (cosine_similarities[i] > 0)]
            for i,each in enumerate(data): 
                if re.search(re_item, each['name'], flags=re.IGNORECASE):
                        filtered_docs_indices.append(i)
                for word in each['name'].split(' '):
                    for look in query.split(' '):
                        if re.search(look, word, flags=re.IGNORECASE):
                            filtered_docs_indices.append(i)
            for i,each in enumerate(data): 
                if re.search(re_item, each['title'], flags=re.IGNORECASE):
                        filtered_docs_indices.append(i)
                for word in each['title'].split(' '):
                    for look in query.split(' '):
                        if re.search(look, word, flags=re.IGNORECASE):
                            filtered_docs_indices.append(i) 

        #consider follower and time
        list_time_follower = []
        list_time = []
        list_follower = []
        for i in filtered_docs_indices:
            list_time.append(data[i]['url'][2])
        for i in filtered_docs_indices:
            list_follower.append(data[i]['followers_count'])

        normed_time = [i/sum(list_time) for i in list_time]
        normed_time = [x * 2 for x in normed_time]
        normed_followers = [i/sum(list_follower) for i in list_follower]


        lst= list(map(operator.add, normed_time,normed_followers))

        order=[]
        for idx, i in enumerate(filtered_docs_indices):
            order.append([i,lst[idx]])

        order= sorted(order, key=itemgetter(1),reverse=True)
        order2 = [item[0] for item in order]
                
        returned = [{"name": master_df['name'].iat[idx],"time" :master_df['time'].iat[idx], "title": (master_df['title'].iat[idx].split("https:",1))[0], "url": master_df['url'].iat[idx]} for idx in order2]
        returned = returned[:8]
        return render_template('index.html', query=query, returned=returned)
@app.route('/', methods=('GET', 'POST'))
def index():
	if request.method == 'POST':
		query = request.form['search_field']
		return redirect(url_for('search', query=query))
	
	return render_template('index.html')


app.secret_key = 'some key that you will never guess'
#Run the app on localhost port 5000
#debug = True -> you don't have to restart flask
#for changes to go through, TURN OFF FOR PRODUCTION
if __name__ == "__main__":
	#app.run('0.0.0.0', 5012, debug = True)
	app.run('127.0.0.1', 5092, debug = True)
	
