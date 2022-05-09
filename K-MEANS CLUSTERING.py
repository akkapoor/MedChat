import pandas as pd
import numpy as np


data=pd.read_excel('data1.xlsx') #Included  data file consisting of articles from Pubumed , this shall be the input to the PEGASUS model used for abstractive summarization
print(data.head())

idea=data.iloc[:,0:1] 
print(data.head())

corpus=[]
for index,row in idea.iterrows():
    corpus.append(row['Idea'])


corpus=data['Idea']  
#Using COuntVectorizer and tdif transformer
#Using libraries from SKlearn

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
corpus=corpus.dropna()
X = vectorizer.fit_transform(corpus)
    

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)                        

from sklearn.cluster import KMeans

num_clusters = 5 # % genetic disorders taken into consideration hence 5 clusters
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf)
clusters = km.labels_.tolist()

idea={'Idea':corpus, 'Cluster':clusters} #Creating dict 
frame=pd.DataFrame(idea,index=[clusters], columns=['Idea','Cluster']) # Converting it into a dataframe.

print("\n")
print(frame['Cluster'].value_counts()) #Print the number of articles belonging to each of the 5 genetic disroders

''' references:
https://github.com/akanshajainn/K-means-Clustering-on-Text-Documents
https://github.com/tsaiian/Documents-Clustering-using-K-Means-Algorithm
'''

