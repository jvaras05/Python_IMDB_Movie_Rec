#!/usr/bin/env python
# coding: utf-8

# # Imports 

# In[1]:


import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from requests import get
from IPython.display import Image
import json
from sklearn.metrics.pairwise import cosine_similarity


# # Class IMDB

# In[2]:


class IMDB:
    def __init__(self, ref_movie): #Inicializando atributos de la clase IMDB
        self._imdb_url= 'https://www.imdb.com'
        self.ref_movie= ref_movie
        self.liked_movie_id=''
        self._rating_value=0
        self._imdb_result_url=''
        self._title=''
    
    def __find_refmovie_imdb(self): #Scraper para buscar la pelicula ingresada en el listado de peliculas de imdb
        print('Conectando con IMDb')
        scraper_search= BeautifulSoup(
        get(f'{self._imdb_url}/find?q={self.ref_movie}&s=tt&exact=true&ref_=fn_tt_ex').text,
        'html.parser')

        #Una vez obtenida la pelicula a evaluar se asignan los valores a los atributos de la clase
    
        imdb_result=scraper_search.find(class_='result_text')
        
        self.liked_movie_id=imdb_result.a["href"].split('/')[2] #asignando id de pelicula
        self._imdb_result_url=imdb_result.a["href"] #asignando url de la pelicula
        self._title=imdb_result.text.strip()  #asginando titulo de la pelicula
        
    def __find_likedmovie_imdb(self): #Scraper para buscar detalles de la pelicula seleccionada del listado anterior
        print(f"Obteniendo Detalle de {self._title}")
        scraper_movie= BeautifulSoup(
        get(f'{self._imdb_url}/{self._imdb_result_url}').text,
        'html.parser')
        self._rating_value=float(scraper_movie.find("span", itemprop="ratingValue").text) #obteniendo rating

        for item in scraper_movie.findAll("div", class_="see-more inline canwrap"):
            if 'Genres' in item.h4.text:
                categorias=[categoria.text.strip() for categoria in item.findAll("a")] #extrayendo categorias de la pelicula

        df_imdb=pd.DataFrame( #armando dataframe de la pelicula con los atributos de la clase
            [{
                'id_movie': self.liked_movie_id,
                'title': self._title,
                'vote_average': self._rating_value,
                'poster_path': '',
                'categories': categorias
            }])
        return df_imdb #retornando dataframe
        
    def get_details_movie_imdb(self): #metodo publico para llamar a los metodos privados de busqueda
            self.__find_refmovie_imdb()
            df_imdb=self.__find_likedmovie_imdb()
            return df_imdb
        

imdb=IMDB(input("Ingrese la pelicula de su preferencia: "))
df_imdb=imdb.get_details_movie_imdb()
df_imdb


# # Class MOVIEDB

# In[3]:


class Moviedb:
    def __init__(self):
        self._api_url='https://api.themoviedb.org/3'
        self._api_key='05350dc485a20a8fb36716869914a328'
    
    def __get_movies(self): #Metodo para obtener las peliculas en cartelera en moviedb
        print("Obteniendo Peliculas en Cartelera...")
        api_now_playing=get(f'{self._api_url}/movie/now_playing?api_key={self._api_key}').text
        now_playing=json.loads(api_now_playing)["results"]
        df_movies=pd.DataFrame(now_playing)[['id', 'title', 'vote_average', 'genre_ids', 'poster_path']]
        return df_movies #retornar peliculas en formato dataframe
    
    def __get_genres(self): #Metodo para obtener listado de generos de peliculas de moviedb
        print("Obteniendo Listado de Generos...")
        api_genres_list=get(f'{self._api_url}/genre/movie/list?api_key={self._api_key}').text
        genres_list=json.loads(api_genres_list)["genres"]
        df_genres=pd.DataFrame(genres_list)
        return df_genres #retornar generos en formato dataframe
    
    def get_cartelera(self): #Metodo para trasponer generos a peliculas y armar cartelera moviedb    
        df_movies=self.__get_movies()
        df_genres=self.__get_genres()
        
        print("Abriendo listado de categorias")
        df_movies_melt=pd.DataFrame({
        col: np.repeat(df_movies[col].values, df_movies['genre_ids'].str.len())
        for col in df_movies.columns.drop('genre_ids')
        }).assign(**{'genre_ids':np.concatenate(df_movies['genre_ids'].values)})[df_movies.columns]
        
        print("Obteniendo Descripciones")
        df_cartelera=df_movies_melt.merge(df_genres, how='inner', left_on='genre_ids', right_on='id')                      .drop(columns=['id_y', 'genre_ids']).rename(columns={'id_x':'id_movie', 'name':'categories'})

        return df_cartelera

moviedb=Moviedb()
moviedb.get_cartelera()
        


# # Class Recomendacion

# In[11]:


class Recomendacion:
    def __init__(self):
        imdb=IMDB(input("Ingrese la pelicula de su preferencia: "))
        self._df_imdb=imdb.get_details_movie_imdb()
        moviedb=Moviedb()
        self._df_cartelera=moviedb.get_cartelera()
        self._titulo_rec=''
        self._url_img=''
        self._ref_movie=imdb.ref_movie
    
    def __merge_imdb_moviedb(self): #Incluir pelicula buscada en IMBD a Cartelera MoviesDB
        print('Incluyendo IMBD a Movies DB')
        df_imdb_melt=pd.DataFrame({
            col: np.repeat(self._df_imdb[col].values, self._df_imdb['categories'].str.len())
            for col in df_imdb.columns.drop('categories')
        }).assign(**{'categories':np.concatenate(df_imdb['categories'].values)})[self._df_imdb.columns]

        df_cartelera_general=self._df_cartelera.append(df_imdb_melt, sort=False).reset_index(drop=True)
        df_cartelera_general['categories']=df_cartelera_general['categories'].map(lambda x: x.replace('Sci-Fi','Science Fiction'))
        return df_cartelera_general
    
    def __matriz_categorias(self): #Metodo para crear matriz de categorias
        print("Verificando categorias por pelicula")
        df_cartelera_general=self.__merge_imdb_moviedb()
        df_cartelera_general.categories.unique()
        
        for item in df_cartelera_general.categories.unique():
            df_cartelera_general[item]=(df_cartelera_general['categories']==item).map({True:1, False:0})

        df_cartelera_table=df_cartelera_general.drop(columns=['categories'], axis=1)
        df_cartelera_table=df_cartelera_table.groupby(['id_movie', 'title', 'vote_average', 'poster_path'], as_index=False).sum()
        return df_cartelera_table
    
    def __matriz_similitud(self): #Metodo para calcular similitud entre peliculas basado en la funcion cosine_similarity
        df_cartelera_table=self.__matriz_categorias()
        print("Creando Matriz dummy")
        mt_dummy=df_cartelera_table.iloc[:,4:]
        
        print("obteniendo similitud")
        similitud = cosine_similarity(mt_dummy, mt_dummy)
        msimil = pd.DataFrame( similitud, columns=df_cartelera_table.id_movie.unique(), index=df_cartelera_table.id_movie.unique())
        return msimil
    
    def __calcular_recomendacion(self): #Metodo para recomendar peliculas con al menos 70% de similitud a la ingresada
        msimil=self.__matriz_similitud()
        df_cartelera_table=self.__matriz_categorias()
        recommended=msimil.index[msimil[imdb.liked_movie_id]>=0.7].to_list()
        recommended.remove(imdb.liked_movie_id)
        df_final=df_cartelera_table[df_cartelera_table['id_movie'].isin(recommended)].sort_values(by=['vote_average'], ascending=False).iloc[:1,:4]
        self._titulo_rec=df_final['title'].to_list()[0]    
        self._url_img='https://image.tmdb.org/t/p/w500'+df_final['poster_path'].to_list()[0]
    
    def get_recomendacion(self): #Desplegar mensaje con recomendacion de pelicula para el usuario
        self.__calcular_recomendacion()
        print (f'Porque viste {self._ref_movie}, te recomendamos: {self._titulo_rec}')
        i=Image(self._url_img, width=200)
        display(i)
    
        
reco=Recomendacion()
reco.get_recomendacion()

