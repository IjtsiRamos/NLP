#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math

#Clase 1

def create_vocabulary(list_of_documents):
    i = 0
    vocabulary = {}
    # Para cada documento en la lista de documentos
    for document in list_of_documents:
        for word in document:
            # Si la palabra no está en el vocabulario
            if word not in vocabulary:
                vocabulary[word] = i
                i += 1
    return vocabulary

def create_vectorized_document(document, vocabulary):
    # Separa el documento en palabras
    # Crea un vector de ceros
    vector = np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in document:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]] += 1
    return vector

def create_matrix_documents(list_of_documents, vocabulary):
    # Crea una matriz de ceros
    matrix = np.zeros((len(list_of_documents), len(vocabulary)))
    # Para cada documento en la lista de documentos
    i=0
    for keys,values in list_of_documents.items():
        # Crea un vector para el documento
        vector = create_vectorized_document(list_of_documents[keys], vocabulary)
        # Agrega el vector a la matriz
        matrix[i] = vector
        i+=1
    return matrix

def calculate_cosine_similarity(vector1, vector2):
    # Multiplica los vectores
    dot_product = np.dot(vector1, vector2)
    # Calcula la norma de los vectores
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    # Calcula la similitud coseno
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def calculate_similarity_for_query(query,documents, matrix_documents, vocabulary):
    # Crea un vector para la consulta
    vector_query = create_vectorized_document(query, vocabulary)
    # Crea una lista para las similitudes
    similarities = {}
    # Para cada documento en la matriz de documentos
    i=0
    for keys,values in documents.items():
        # Calcula la similitud coseno
        similarity = calculate_cosine_similarity(vector_query, matrix_documents[i])
        # Agrega la similitud a la lista de similitudes
        similarities[keys] = similarity
        i+=1
    return sort_dictionary_by_value(similarities)

def sort_dictionary_by_value(dictionary):
    # Ordena el diccionario
    sorted_dictionary = {k: v for k,v in sorted(dictionary.items(), key=lambda x: x[1], reverse=True)}
    return sorted_dictionary



#Metodo de ranking

def create_vocabulary2(list_of_documents):
    i = 0
    vocabulary = {}
    # Para cada documento en la lista de documentos
    for document_no,document in list_of_documents.items():
        for word in document:
            # Si la palabra no está en el vocabulario
            if word not in vocabulary:
                vocabulary[word] = {'id':i, 'docto':[document_no]}
                i += 1
            else:
                vocabulary[word]['docto'].append(document_no)
    for word in vocabulary:
        vocabulary[word]['df']= len(set(vocabulary[word]['docto']))
                
    return vocabulary

def create_vectorized_document2(document, vocabulary, n_docs):
    # Separa el documento en palabras
    # Crea un vector de ceros
    vector = np.zeros(len(vocabulary))
    ivector= np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in document:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]['id']] += 1
    for word in vocabulary:
        ivector[vocabulary[word]['id']]= n_docs/vocabulary[word]['df']
            
    return (np.log(np.array(vector)+1))*ivector


def create_matrix_documents2(list_of_documents, vocabulary):
    # Crea una matriz de ceros
    matrix = np.zeros((len(list_of_documents), len(vocabulary)))
    i=0
    # Para cada documento en la lista de documentos
    for keys,values in list_of_documents.items():
        # Crea un vector para el documento
        vector = create_vectorized_document2(list_of_documents[keys], vocabulary,len(list_of_documents))
        # Agrega el vector a la matriz
        matrix[i] = vector
        i+=1
    return matrix
    
    

def calculate_similarity_for_query2(query,documents, matrix_documents, vocabulary):
    # Crea un vector para la consulta
    vector_query = create_vectorized_document2(query, vocabulary, len(documents))
    # Crea una lista para las similitudes
    similarities = {}
    # Para cada documento en la matriz de documentos
    i=0
    for keys,values in documents.items():
        # Calcula la similitud coseno
        similarity = calculate_cosine_similarity(vector_query, matrix_documents[i])
        # Agrega la similitud a la lista de similitudes
        similarities[keys] = similarity
        i+=1
    return sort_dictionary_by_value(similarities)

#Metodo Probabilistico

def create_vocabulary3(list_of_documents):
    i = 0
    vocabulary = {} # Diccionario para almacenar las palabras y el número de palabra
    n = len(list_of_documents)
    # Para cada documento en la lista de documentos
    for document_no, document in list_of_documents.items():
        # Separa el documento en palabras
        # Para cada palabra en las palabras
        for word in document:
            # Si la palabra no está en el vocabulario
            # Agregamos el ID y el documento que aparece
            if word not in vocabulary:
                vocabulary[word] = {'id':i, 'docto':[document_no]} # {ID:número de palabra, DOCTO:número de documento}
                i += 1
            # En otro caso solo agregamos el documento en el que aparece
            else:
                vocabulary[word]['docto'].append(document_no)
        
        for word in vocabulary:
            # df (document frequency) que es el número de documentos que contiene cada palabra
            vocabulary[word]['df'] = ((n+0.5) /(len(vocabulary[word]['docto'])))
    return vocabulary

def create_vectorized_document3(document,vocabulary, n_docs, mu, sigma):
    # Separa el documento en palabras
    # Crea un vector de ceros
    vector = np.zeros(len(vocabulary))
    ivector= np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in  document:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]['id']] += 1
    for word in vocabulary:        
            ivector[vocabulary[word]['id']]= n_docs/vocabulary[word]['df']
    L = abs((len(document) - mu)/sigma)
    tf = np.array(vector) / (np.array(vector) + 0.5 + 1.5 * L)
    idf = np.log((np.array(ivector))/np.log(n_docs+1))
    # Con un alpha = 0.4 y beta = 0.6
    return 0.4 *0.6 * tf * idf 

def create_matrix_documents3(list_of_documents, vocabulary):
    # Crea una matriz de ceros
    size=[]
    for document_no, document in list_of_documents.items():
        size.append(len(document))
    n=len(list_of_documents)
    mu= sum(size)/n
    sigma = (1/n)*sum([(x-mu)*2 for x in size])
    matrix = np.zeros((len(list_of_documents),len(vocabulary)))
    i=0
    # Para cada documento en la lista de documentos
    for keys,values in list_of_documents.items():
        # Crea un vector para el documento
        vector = create_vectorized_document3(list_of_documents[keys],vocabulary, len(list_of_documents), mu, sigma)
        # Agrega el vefctor a la matriz
        matrix[i] = vector
        i+=1
    return matrix  

def calculate_similarity_for_query3(query,documents, matrix_documents, vocabulary):
    size=[]
    for document_no, document in documents.items():
        size.append(len(document))
    n=len(documents)
    mu= sum(size)/n
    sigma = (1/n)*sum([(x-mu)*2 for x in size])    
    # Crea un vector para la consulta
    vector_query = create_vectorized_document3(query, vocabulary, len(documents),mu,sigma)

    # Crea una lista para las similitudes
    similarities = {}
    # Para cada documento en la matriz de documentos
    i=0
    for keys,values in documents.items():
        # Calcula la similitud coseno
        similarity = calculate_cosine_similarity(vector_query, matrix_documents[i])
        # Agrega la similitud a la lista de similitudes
        similarities[keys] = similarity
        i+=1
    return sort_dictionary_by_value(similarities)   

   
    
# Metodo Probabilistico 2

def create_vocabulary4(list_of_documents):
    i = 0
    vocabulary = {} # Diccionario para almacenar las palabras y el número de palabra
    n = len(list_of_documents)
    # Para cada documento en la lista de documentos
    for document_no, document in list_of_documents.items():
        # Separa el documento en palabras
        # Para cada palabra en las palabras
        for word in document:
            # Si la palabra no está en el vocabulario
            # Agregamos el ID y el documento que aparece
            if word not in vocabulary:
                vocabulary[word] = {'id':i, 'docto':[document_no]} # {ID:número de palabra, DOCTO:número de documento}
                i += 1
            # En otro caso solo agregamos el documento en el que aparece
            else:
                vocabulary[word]['docto'].append(document_no)
        
        for word in vocabulary:
            # df (document frequency) que es el número de documentos que contiene cada palabra
            vocabulary[word]['df'] = (n - len(vocabulary[word]['docto']) + 0.5)/(len(vocabulary[word]['docto']) + 0.5)
    return vocabulary

def create_vectorized_document4(document,vocabulary, n_docs, mu, sigma):
    # Separa el documento en palabras
    # Crea un vector de ceros
    result= np.zeros(len(vocabulary))
    vector = np.zeros(len(vocabulary))
    ivector= np.zeros(len(vocabulary))
    # Para cada palabra en las palabras
    for word in  document:
        # Si la palabra está en el vocabulario
        if word in vocabulary:
            # Incrementa el valor del vector en la posición de la palabra
            vector[vocabulary[word]['id']] += 1
    L = abs((len(document) - mu)/sigma)
    n = n_docs
    for word in vocabulary:
        if 2*len(vocabulary[word]['docto'])>= n:
            result[vocabulary[word]['id']] =1
        else:
            result[vocabulary[word]['id']] =((vector[vocabulary[word]['id']]*(1.5+1)) / (vector[vocabulary[word]['id']] + 1.5 * L)) * np.log(vocabulary[word]['df']+1)
    # Con un alpha = 0.4 y beta = 0.6
    return  result

def create_matrix_documents4(list_of_documents, vocabulary): 
    # Crea una matriz de ceros
    size=[]
    for document_no, document in list_of_documents.items():
        size.append(len(document))
    n=len(list_of_documents)
    mu= sum(size)/n
    sigma = (1/n)*sum([(x-mu)*2 for x in size])
    matrix = np.zeros((len(list_of_documents),len(vocabulary)))
    i=0
    # Para cada documento en la lista de documentos
    for keys,values in list_of_documents.items():
        # Crea un vector para el documento
        vector = create_vectorized_document4(list_of_documents[keys],vocabulary, len(list_of_documents), mu, sigma)
        # Agrega el vefctor a la matriz
        matrix[i] = vector
        i+=1
    return matrix  
def calculate_similarity_for_query4(query,documents, matrix_documents, vocabulary):
    size=[]
    for document_no, document in documents.items():
        size.append(len(document))
    n=len(documents)
    mu= sum(size)/n
    sigma = (1/n)*sum([(x-mu)*2 for x in size])    
    # Crea un vector para la consulta
    vector_query = create_vectorized_document4(query, vocabulary, len(documents),mu,sigma)

    # Crea una lista para las similitudes
    similarities = {}
    # Para cada documento en la matriz de documentos
    i=0
    for keys,values in documents.items():
        # Calcula la similitud coseno
        similarity = calculate_cosine_similarity(vector_query, matrix_documents[i])
        # Agrega la similitud a la lista de similitudes
        similarities[keys] = similarity
        i+=1
    return sort_dictionary_by_value(similarities)   


# In[ ]:




