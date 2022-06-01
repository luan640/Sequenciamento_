import datetime
from datetime import timedelta
import pandas as pd
from fbprophet import Prophet
from sklearn import metrics
from fbprophet.plot import add_changepoints_to_plot
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import performance_metrics
from fbprophet.diagnostics import cross_validation
from fbprophet.plot import plot_plotly, plot_components_plotly
from math import sqrt
import streamlit as st
from plotly import graph_objs as go

st.title('Previsão de vendas por modelo de carreta')

arquivo = 'BASE VENDAS ATUALIZADA.xlsx'

df = pd.read_excel(arquivo, parse_dates=['ds'])
df['MOD08'].unique()#ordenando valores por data

lista_carretas = df['MOD08'].unique().tolist()
lista_carretas = str(lista_carretas)[1:-1]

#criando sidebar

with st.sidebar:
    modelo_carreta = st.selectbox('Escolha um modelo de carreta', ('CBHM','CBH','F','FTC','FA','P.A','ROBUSTA'))
    meses = st.slider('Meses a serem previsto',1,12)
    
df = df.sort_values(by='ds')

#selecionando modelo de carreta

selecao = (df.MOD08 == modelo_carreta)
df1 = df[selecao]

#deixando apenas colunas necessárias

df1 = df1[['ds','y']]

#definindo data como index

df1.set_index('ds', inplace=True)

df1 = df1.resample('m').sum()

treino = df1[df1.index <= pd.to_datetime("2021-12-31", format='%Y-%m-%d')]
teste = df1[df1.index >= pd.to_datetime("2022-01-31", format='%Y-%m-%d')]

treino = treino.reset_index()
teste = teste.reset_index()

modelo = Prophet(yearly_seasonality=True)
modelo.fit(treino)

df_previsoes = pd.DataFrame({'ds': teste['ds'].values})

pred = modelo.predict(df_previsoes)

#erro médio do modelo
from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(teste.y, pred.yhat))

perf = modelo.plot(pred, figsize=(30,10))

df1 = df1.reset_index()

modelo = Prophet()
modelo.fit(df1)

futuro = modelo.make_future_dataframe(periods = meses, freq = 'M')

pred = modelo.predict(futuro)

#fig1 = modelo.plot(pred, figsize=(10,5))

st.subheader("Previsão")
grafico1 = plot_plotly(modelo, pred)
st.plotly_chart(grafico1)

st.subheader("Tendências")
grafico2 = plot_components_plotly(modelo, pred)
st.plotly_chart(grafico2)

plt.figure(figsize=(25,10))
plt.plot(df1.set_index('ds')['y'], color='black', label = 'treino')
plt.plot(pred.set_index('ds')['yhat'], color='green', label = 'previsto')
plt.legend(['treino', 'previsto'], loc=1)
plt.title('Valores de treino e previsto') # Título do gráfico

#gráfico

st.subheader("Valores reais VS previsto")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df1['ds'],
                         y=df1['y'],
                         name='Valores reais',
                         line_color='black', mode='lines+markers'))

fig.add_trace(go.Scatter(x=pred['ds'],
                         y=pred['yhat'],
                         name='Valores previsto',
                         line_color='green', mode='lines+markers'))

st.plotly_chart(fig)


pred_new = pred[['ds','yhat']]
pred_new = pred_new[:len(df1)]

try:
    rmse_new = sqrt(mean_squared_error(df1.y, pred_new.yhat))

except:
    pass
