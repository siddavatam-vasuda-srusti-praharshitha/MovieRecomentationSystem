#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import ast as ast


# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')


# In[4]:


movies= movies.merge(credits,on='title')


# In[5]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[6]:


movies.info()


# In[7]:


movies.head()


# In[8]:


movies.isnull().sum()


# In[9]:


movies.dropna(inplace=True)


# In[10]:


movies.duplicated().sum()


# In[11]:


movies.iloc[0].genres


# In[12]:


def convert(obj):
	L=[]
	for i in ast.literal_eval(obj):
		L.append(i['name'])
	return L
movies['genres'] = movies['genres'].apply(convert)


# In[13]:


movies.head()


# In[14]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[15]:


#print(movies['cast'].dtypes)


# In[16]:


def convert3(obj):
    L=[]
    counter=0
    data = ast.literal_eval(obj)
    for i in data:
        if counter != 3 :
            L.append(i['name'])
            counter+=1
            print(counter)
        else:
            break
    return L


# In[17]:


movies['cast']= movies['cast'].apply(convert3)


# In[ ]:





# In[18]:


movies.head()


# In[19]:


movies['crew'][0]


# In[20]:


def fetch_director(obj):
    L=[]
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[21]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[22]:


movies.head()


# In[23]:


def returnList(obj):
    return obj.split()
movies['overview'] = movies['overview'].apply(returnList)


# In[24]:


movies.head()


# In[25]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[26]:


movies.head()


# In[27]:


movies['tags']=movies['overview']+ movies['genres']+ movies['keywords']+ movies['cast']+ movies['crew']


# In[28]:


movies.head()


# In[29]:


new_df = movies[['movie_id','title','tags']]


# In[30]:


new_df['tags']= new_df['tags'].apply(lambda x:" ".join(x))


# In[31]:


new_df.head()


# In[32]:


get_ipython().system('pip install nltk')


# In[33]:


import nltk


# In[34]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[35]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
         


# In[36]:


new_df['tags']=new_df['tags'].apply(stem)


# In[37]:


new_df['tags'][0]


# In[38]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[39]:


new_df.head()


# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[41]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[42]:


vectors


# In[43]:


vectors[0]


# In[44]:


cv.get_feature_names()


# In[45]:


from sklearn.metrics.pairwise import cosine_similarity


# In[46]:


similarity=cosine_similarity(vectors)


# In[47]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[48]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances=similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[49]:


recommend('Batman Begins')


# In[50]:


import pickle


# In[51]:


pickle.dump(new_df,open('D:\\ML_Project\\movies.pkl','wb'))
#f = open("movies.pkl", "wb")
#f.write(new_df)
#f.close()


# In[52]:


new_df['title'].values


# In[53]:


pickle.dump(new_df.to_dict(),open('D:\\ML_Project\\movie_dict.pkl','wb'))


# In[54]:


pickle.dump(similarity,open('D:\\ML_Project\\similarity.pkl','wb'))


# In[55]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[56]:


pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:





# In[ ]:




