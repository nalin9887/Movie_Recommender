import numpy as np
import pandas as pd

movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')

movies['keywords'].head()
movies=movies.merge(credits,on='title')

movies.head()

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


movies.head()
import ast
#To convert string to list 
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L 
    
def convert_top_3(text):
    L = []
    counter=0
    for i in ast.literal_eval(text):
        if counter!=3:
            L.append(i['name']) 
            counter+=1
        else:
            break
    return L 
def find_director(text):
    L = []  
    for i in ast.literal_eval(text):    
        if i['job'] == 'Director':  
            L.append(i['name'])
            break 
    return L     
movies.dropna(inplace=True)
movies['genres'] = movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
movies['cast'] = movies['cast'].apply(convert_top_3)
movies.head()   
movies['crew']=movies['crew'].apply(find_director)
movies.head()
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies.head()


movies['genres']=movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies.head()
movies['tags']=movies['genres']+movies['crew']+movies['cast']+movies['keywords']+movies['overview']
movies.head()

new_df=movies[['movie_id', 'title', 'tags']]
new_df.head()
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
new_df.head()   #                           
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())
new_df.head()    


from nltk.stem.porter import PorterStemmer  # remove similar word example : loved , loveing to love ,love
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
       y.append(ps.stem(i))
    return " ".join(y)

new_df['tags']=new_df['tags'].apply(stem)

new_df.head()

from sklearn.feature_extraction.text import CountVectorizer
cv =CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)


def recommend(movies):
    movie_index = new_df[new_df['title']==movies].index[0]
    recommended_movie=sorted(list(enumerate(similarity[movie_index])),reverse=True,key=lambda x:x[1])[1:6]
    for i in recommended_movie:
        print(new_df.iloc[i[0]]['title'])

recommend("Avatar")

import pickle 

pickle.dump(open('movie_list.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
