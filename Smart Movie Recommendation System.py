import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate
from nltk.stem.snowball import SnowballStemmer
import warnings

warnings.simplefilter('ignore')

"""


**Part I**

Content Based Recommender**

File **metadata.csv** contains all information about movies in the dataset.
"""

metadata = pd.read_csv("metadata.csv")
metadata = metadata.drop(['Unnamed: 0'], axis=1)
metadata.genres = metadata.genres.fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
metadata['year'] = pd.to_datetime(metadata['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
metadata.head()

"""
File **links_small.csv** contains related movies ids from other files. But only *tmdbId* is needed.
As a result, we get **9099** films, that are in metadata dataframe.
Let **avail** dafaframe be metadata films, that are available to process in the next step
"""

links_small = pd.read_csv('links_small.csv')
links_small = links_small[links_small.tmdbId.notnull()].tmdbId
links_small = links_small.astype('int64')
links_small.head(7)

avail = metadata[metadata.id.isin(links_small)]
len(avail)



avail.tagline = avail.tagline.fillna('')
avail.overview = avail.overview.fillna('')
avail['description'] = avail.tagline + " " + avail.overview

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(avail.description)
print(tfidf_matrix.shape)

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[1]

titles = avail.title
indices = pd.Series([i for i in range(len(avail))], index=avail.title)

def get_recommendations(title, number):
    try:
        idx = indices[title]
    except:
        print("Film (%s) does not exist in the dataset" % title)
        return
    
    if type(idx) != np.dtype('int64') and len(idx) > 1:
        print("There are several films called (%s)" % title)
        print("Their indices are: ", avail[avail.title == title].index)
        idx = sorted(idx, key=lambda x: avail.iloc[x].popularity, reverse=True)
        idx = idx[0]
        print("For recommendation, I will take the most popular one with id ", avail.iloc[idx].id)

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:number+1]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]

"""
Now, let's get 10 recommendations for the film **The Terminator**
"""

get_recommendations('The Terminator', 10)

"""
If the input film is not in the dataframe, corresponding error is raised.

Sometimes happens, that some films have similar titles. In that case, I choose the most popular one. 
For example, **Titanic** movie:
"""

get_recommendations('Titanic', 10)

"""
**Recommendations based on cast, crew, keywords and genres**

At first, merge **credits** and **keywords** dataframes with our **metadata**. 
"""

credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')

metadata = metadata.merge(credits, on='id').merge(keywords, on='id')
metadata = metadata.drop(['Unnamed: 0'], axis=1)
metadata.head()

"""
Then, reassign the value of **avail**, available films. Now, it's **9663** movies.
"""

avail = metadata[metadata.id.isin(links_small)]
print(len(avail))

"""

From crew, we will only pick a director of the movie as a feature since others don't contribute that much to the feel of the movie. As a result, for a director to have more influence then for a regular keyword, for example, repeate a director 3 times.
Keywords will be converted to the lists of stemmed words.We will take only keywords that have occured more then 3 times in the dataset.
From cast, we will take only three main actors.
"""

avail['cast'] = avail['cast'].apply(literal_eval)
avail['keywords'] = avail['keywords'].apply(literal_eval)
avail['crew'] = avail['crew'].apply(literal_eval)
avail['cast_size'] = avail['cast'].apply(lambda x: len(x))
avail['crew_size'] = avail['crew'].apply(lambda x: len(x))

def get_director(crew):
    for member in crew:
        if member['job'] == 'Director':
            return member['name']
    return np.nan    
        
avail['director'] = avail['crew'].apply(get_director)
#avail['cast'] = avail['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
avail['cast'] = avail['cast'].apply(lambda x: x[:3] if len(x) > 3 else x)
avail['keywords'] = avail['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])

avail.cast = avail.cast.apply(lambda x: [w.replace(" ", "").lower() for w in x])
avail.director = avail.director.astype('str').apply(lambda x: x.replace(" ", "").lower())
avail.director = avail.director.apply(lambda x: [x, x, x])

avail.head()

"""
Keywords preprocessing
"""

key = avail.apply(lambda x: pd.Series(x.keywords), axis=1).stack().reset_index(level=1, drop=True)
key.name = 'keyword'
key = key.value_counts()
print(key[:10])
key = key[key > 3]

stemmer = SnowballStemmer('english')
stemmer.stem('films')

def filter_keywords(keywords):
    words = []
    for i in keywords:
        if i in key:
            words.append(i)
    return words        

avail.keywords = avail.keywords.apply(filter_keywords)
avail.keywords = avail.keywords.apply(lambda x: [stemmer.stem(i) for i in x])
avail.keywords = avail.keywords.apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])

avail['stack'] = avail.keywords + avail.cast + avail.genres + avail.director
avail['stack'] = avail['stack'].apply(lambda x: ' '.join(x))
avail.head()

"""
We now have our cast, director, keywords and genres stacked in the **stack** column.
We will use the same Cosine Similarity, but now with CountVecrorizer matrix.
"""

count = CountVectorizer(analyzer='word',ngram_range=(1, 3),min_df=0)
count_matrix = count.fit_transform(avail['stack'])

cosine_sim = linear_kernel(count_matrix, count_matrix)

titles = avail.title
indices = pd.Series([i for i in range(len(avail))], index=avail.title)

"""
Let's get recommendations for **The Terminator** film. As you can see, we have **Avatar** and **Titanic** films as recommendations. That's because thay all have a same director: James Kameron. As for me, this system works better than previous one. So I will use these cosine_sim values in the next Hybrid system
"""

print(get_recommendations('The Terminator', 10))

print(get_recommendations('Rocky', 10))

"""
**Part II**

**Collaborative Filtering**

"""

"""
In this part, We will use a technique called **Collaborative Filtering** to make recommendations. Collaborative Filtering is based on the idea that users similar to me can be used to predict how much we will like a particular product or service those users have experienced but I have not.

For that We will use the **Surprise** library that used extremely powerful algorithms like **Singular Value Decomposition (SVD)** to minimise RMSE (Root Mean Square Error) and give great recommendations.

**Surpsise** is a Python scikit building and analysing recommender systems.
"""

reader = Reader()
ratings = pd.read_csv("ratings_small.csv")
ratings.head()

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
print(data.df.head())
#data.split(n_folds=5)

svd = SVD(n_factors=100, n_epochs=40, lr_all=0.005, reg_all=0.2, verbose=False)
cross_validate(svd, data, measures=['RMSE', 'MAE'])

"""
After training, we get RMSE = **0.8901**, which is good for our model.
"""

trainset = data.build_full_trainset()
svd.fit(trainset)

"""
**Part III**

**Hybrid system**
"""

"""
In this part, we will build hybrid system. How model works: get 50 top scoring films from the cosine_sim matrix; for a particular user, sort them by predicted rating for user.
"""

def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
    
id_map = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]
id_map.tmdbId = id_map.tmdbId.apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(avail[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')

def hybrid(userId, title, number=10):
    try:
        idx = indices[title]
    except:
        print("Film (%s) does not exist in the dataset" % title)
        return
    
    if type(idx) != np.dtype('int64') and len(idx) > 1:
        print("There are several films called (%s)" % title)
        print("Their indices are: ", avail[avail.title == title].index)
        idx = sorted(idx, key=lambda x: avail.iloc[x].popularity, reverse=True)
        idx = idx[0]
        print("For recommendation, I will take the most popular one with id ", avail.iloc[idx].id)
        
    tmdbId = id_map.loc[title]['id']
    movie_id = id_map.loc[title]['movieId']    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:50]
    
    movie_indices = [i[0] for i in sim_scores]
    movies = avail.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est * 2)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(number)

hybrid(34, 'Inception', 10)

hybrid(10, 'Inception', 10)

hybrid(44, 'Alien', 10)

"""
We see that for our hybrid recommender, we get different recommendations for different users although the movie is the same.
"""

