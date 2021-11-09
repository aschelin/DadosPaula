# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 16:29:36 2021

@author: asche
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns


df = pd.read_csv('dadospaula_new.csv')
st.title('Entrance Skin Dose Evaluation in Pediatric Patients')
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# SIDEBAR
# Parâmetros e número de ocorrências
tabela = st.sidebar.empty()    # placeholder que só vai ser carregado com o df_filtered

st.sidebar.header("Parâmetros")
info_sidebar = st.sidebar.empty()    # placeholder, para informações filtradas que só serão carregadas depois

selectbox1 = 'pelve'
selectbox2 = 'DR2'
selectbox3 = 'Matriz de Correlação'
selectbox4 = 'Todas as Idades'
selectbox5 = 'Todos os Pesos'

selectbox1 = st.sidebar.selectbox(
    "Tipo de Exame",
    ('torax','pelve','cranio','seios','abdome')
)

selectbox2 = st.sidebar.selectbox(
    "Equipamento",
    ('DR1','DR2')
)

selectbox3 = st.sidebar.selectbox(
    "Tipo de Gráfico",
    ('Matriz de Correlação','Regressão Linear','Scatter Plot')
)

selectbox4 = st.sidebar.selectbox(
    "Idade",
    ('Todas as Idades','Menores que 1','Entre 1 e 5','Entre 5 e 10','Maiores que 10')
)

selectbox5 = st.sidebar.selectbox(
    "Peso",
    ('Todos os Pesos','Menores que 5kg','Entre 5kg e 15kg','Entre 15kg e 30kg','Maiores que 30kg')
)


    
dfselecionado = df.loc[(df['TIPO']==selectbox1) & (df['EQUIPAMENTO ']==selectbox2)]

if (selectbox4 == 'Todas as Idades'):
    dfselecionado = dfselecionado

if (selectbox4 == 'Menores que 1'):
    dfselecionado = dfselecionado[dfselecionado['IDADE (ANOS)']<1]

if (selectbox4 == 'Entre 1 e 5'):
    dfselecionado = dfselecionado[(dfselecionado['IDADE (ANOS)']>1)&(dfselecionado['IDADE (ANOS)']<=5)]

if (selectbox4 == 'Entre 5 e 10'):
    dfselecionado = dfselecionado[(dfselecionado['IDADE (ANOS)']>5)&(dfselecionado['IDADE (ANOS)']<=10)]

if (selectbox4 == 'Maiores que 10'):
    dfselecionado = dfselecionado[(dfselecionado['IDADE (ANOS)']>10)]


if (selectbox5 == 'Todos os Pesos'):
    dfselecionado = dfselecionado

if (selectbox5 == 'Menores que 5kg'):
    dfselecionado = dfselecionado[dfselecionado['PESO (kg)']<1]

if (selectbox5 == 'Entre 5kg e 15kg'):
    dfselecionado = dfselecionado[(dfselecionado['PESO (kg)']>5)&(dfselecionado['PESO (kg)']<=15)]

if (selectbox5 == 'Entre 15kg e 30kg'):
    dfselecionado = dfselecionado[(dfselecionado['PESO (kg)']>15)&(dfselecionado['PESO (kg)']<=30)]

if (selectbox5 == 'Maiores que 30kg'):
    dfselecionado = dfselecionado[(dfselecionado['PESO (kg)']>30)]

st.write('Mostrando o gráfico para:',selectbox1)
st.write('Equipamento:',selectbox2)

if (selectbox3 == 'Regressão Linear'):
    
    dfregressao = dfselecionado[['DAP','DOSE (mGy)']]
    dfregressao.dropna(inplace=True)
    x = dfregressao['DAP'].values.reshape(-1,1)
    y = dfregressao['DOSE (mGy)'].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.7)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_new = model.predict(X_test)
    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    ax.plot(X_test, y_new,color='red')
    ax.set_xlabel('DAP')
    ax.set_ylabel('DOSE')
    r2 = model.score(X_train, y_train)
    st.write('R^2 é igual a',r2)
    st.pyplot(fig)

if (selectbox3 == 'Matriz de Correlação'):
#    if (selectbox1 == 'cranio'):
#        dfselecionado.drop('espessura',axis=1, inplace=True)
#        dfselecionado.drop('dfp',axis=1, inplace=True)
    
    dfcorr = dfselecionado.corr()
    # Generate a mask for the upper triangle
    sns.set_context("talk", font_scale=.7)
    mask = np.triu(np.ones_like(dfcorr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 10))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(dfcorr, mask=mask, cmap=cmap,  vmax=1, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7},annot=True);
    st.pyplot(f)

if (selectbox3 == 'Scatter Plot'):
    sns.set_context("talk", font_scale=1.1)
    fig, ax = plt.subplots(figsize=(12,4))
    sns.scatterplot(x="DAP",
    y="DOSE (mGy)",
    size="PESO (kg)",
    sizes=(20,800),
    alpha=0.5,
    hue="GÊNERO",
    data=dfselecionado)
    # Put the legend out of the figure
    ax.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
    # Put the legend out of the figure
    #plt.legend(bbox_to_anchor=(1.01, 0.54),  borderaxespad=0.)
    ax.set_xlabel('DAP')
    ax.set_ylabel('DOSE')
    st.pyplot(fig)

if tabela.checkbox("Mostrar tabela de dados"):
    st.write(dfselecionado)
    
    

