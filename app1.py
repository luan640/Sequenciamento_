
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



st.title('Previsão de vendas por modelo de carreta')



arquivo = 'BASE VENDAS ATUALIZADA.xlsx'
df = pd.read_excel(arquivo, parse_dates=['ds'])

#selecionando modelo de carreta

modelo_carreta = df['MOD08'].unique().tolist()
modelo_carreta = str(modelo_carreta)[1:-1]

with st.sidebar:
    modelo_carreta = st.selectbox('Escolha um modelo de carreta', ('CBHM','CBH','F','FTC','FA','P.A','ROBUSTA'))
    
selecao = (df.MOD08 == modelo_carreta)
df1 = df[selecao]

#deixando apenas colunas necessárias

df1 = df1[['ds','y']]
df1 = df1.sort_values(by='ds')

df1['ds'] = df1.ds
df1['month'] = df1['ds'].dt.strftime('%B')
df1['year'] = df1['ds'].dt.strftime('%Y')
df1['dayofweek'] = df1['ds'].dt.strftime('%A')
df1['dayofyear'] = df1['ds'].dt.dayofyear
df1['dayofmonth'] = df1['ds'].dt.day
df1['weekofyear'] = df1['ds'].dt.weekofyear

fig1,(ax1,ax2,ax3)= plt.subplots(ncols = 3)
fig1.set_size_inches(40,5)

monthAggregated1 = pd.DataFrame(df1.groupby("month")["y"].sum()).reset_index().sort_values('y')
monthAggregated2 = pd.DataFrame(df1.groupby("dayofweek")["y"].sum()).reset_index().sort_values('y')
monthAggregated3 = pd.DataFrame(df1.groupby("year")["y"].sum()).reset_index().sort_values('y')

st.subheader("Informações sobre o modelo de carreta")

fig1 = px.bar(monthAggregated1, x='month', y='y',title='Vendas por mês: ' + modelo_carreta)
fig2 = px.bar(monthAggregated2, x='dayofweek', y='y',title='Vendas por dia da semana: ' + modelo_carreta)
fig3 = px.bar(monthAggregated3, x='year', y='y',title='Vendas por ano: ' + modelo_carreta )

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)


df1 = df1.set_index('ds')
df1 = df1.resample('M').sum()
df1 = df1.reset_index()

valor = df1['y']

Q1 = valor.quantile(.25)
Q3 = valor.quantile(.75)
IIQ = Q3 - Q1
limite_inferior = Q1 - 1.5 * IIQ
limite_superior = Q3 + 1.5 * IIQ

selecao = (valor >= limite_inferior) & (valor <= limite_superior)
df1 = df1[selecao]

Month = [i.month for i in df1['ds']]
Year = [i.year for i in df1['ds']]

X = np.array([Month, Year]).T

Y_month = [6,7,8,9,10,11,12]
Y_year = [2022,2022,2022,2022,2022,2022,2022]
Y_index = [65,66,67,68,69,70,71]

Y = np.array([Y_month, Y_year,Y_index]).T
data_df = pd.DataFrame(Y)
data_df = data_df.set_index(2)
Y = np.array([Y_month, Y_year]).T

 
# fit the model
my_rf = RandomForestRegressor()
my_rf.fit(X, df1.y.values)
 
# predict on the same period
preds = my_rf.predict(X)
preds1 = my_rf.predict(Y)
 
# plot what has been learned

st.subheader("Previsão com o modelo 1")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df1.index,
                         y=df1.y.values,
                         name='Valores reais',
                         line_color='black', mode='lines+markers'))

fig4.add_trace(go.Scatter(x=df1.index,
                         y=preds,
                         name='Valores previsto',
                         line_color='green', mode='lines+markers'))

fig4.add_trace(go.Scatter(x=data_df.index,
                         y=preds1,
                         name='Valores previsto',
                         line_color='red', mode='lines+markers'))

st.plotly_chart(fig4)

preds1

# fit the model
my_xgb = xgb.XGBRegressor()
my_xgb.fit(X, df1.y.values)
 
# predict on the same period
preds = my_xgb.predict(X)
preds1 = my_xgb.predict(Y)

# plot what has been learned
st.subheader("Previsão com o modelo 2")

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df1.index,
                         y=df1.y.values,
                         name='Valores reais',
                         line_color='blue', mode='lines+markers'))

fig5.add_trace(go.Scatter(x=df1.index,
                         y=preds,
                         name='Valores previsto',
                         line_color='green', mode='lines+markers'))

fig5.add_trace(go.Scatter(x=data_df.index,
                         y=preds1,
                         name='Valores previsto',
                         line_color='red', mode='lines+markers'))

st.plotly_chart(fig5)
preds1
