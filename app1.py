import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

st.title('Previsão de vendas por modelo de carreta')

monet_quant = st.selectbox('Monetário ou unidades vendidas?', ('Monetário','Unidade'))   

if monet_quant == 'Unidade':
    arquivo = 'BASE VENDAS ATUALIZADA_quant.xlsx'
else:
    arquivo = 'BASE VENDAS ATUALIZADA_monet.xlsx'
   
df = pd.read_excel(arquivo) #, parse_dates=['ds'])
    
#selecionando modelo de carreta

modelo_carreta = df['MOD08'].unique().tolist()
modelo_carreta = str(modelo_carreta)[1:-1] 

with st.sidebar:
    modelo_carreta = st.selectbox('Escolha um modelo de carreta', ('CBHM','CBH','F','FTC','FA','P.A','ROBUSTA','GERAL'))
    tratamento_ou_nao = st.selectbox('Para tratamento de outliers', ('Sim','Não'))
    
selecao = (df.MOD08 == modelo_carreta)
df1 = df[selecao]
df1['ds'] = pd.to_datetime(df1['ds'],format='%d/%m/%Y')

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

monthAggregated1 = pd.DataFrame(df1.groupby("month")["y"].sum()).reset_index().sort_values('y')
monthAggregated2 = pd.DataFrame(df1.groupby("dayofweek")["y"].sum()).reset_index().sort_values('y')
monthAggregated3 = pd.DataFrame(df1.groupby("year")["y"].sum()).reset_index().sort_values('y')

st.subheader("Dados históricos de vendas")

fig1 = px.bar(monthAggregated1, x='month', y='y',title='Mes que mais vendeu: ' + modelo_carreta)
fig2 = px.bar(monthAggregated2, x='dayofweek', y='y',title='Dia da semana que mais vendeu: ' + modelo_carreta)
fig3 = px.bar(monthAggregated3, x='year', y='y',title='Ano que mais vendeu: ' + modelo_carreta )

st.plotly_chart(fig1)
st.plotly_chart(fig2)
st.plotly_chart(fig3)

df1 = df1.set_index('ds')
df1 = df1.resample('M').sum()
df1 = df1.reset_index()

#tratamento de informações

Y_index = []

if tratamento_ou_nao == 'Sim':

    valor = df1['y']
    
    Q1 = valor.quantile(.25)
    Q3 = valor.quantile(.75)
    IIQ = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IIQ
    limite_superior = Q3 + 1.5 * IIQ
    
    selecao = (valor >= limite_inferior) & (valor <= limite_superior)
    df1 = df1[selecao]
    
    df1 = df1.reset_index(drop=True)
    
    tamanho_ = df1.shape[0]
    Y_index = np.arange(tamanho_,tamanho_ + 7) 

else:
    
    df1 = df1.reset_index(drop=True)
    tamanho_ = df1.shape[0]
    Y_index = np.arange(tamanho_, tamanho_ + 7) 

fig6 = px.box(df1, y='y', title='BoxPlot')
st.plotly_chart(fig6)
# final do tratamento

Month = [i.month for i in df1['ds']]
Year = [i.year for i in df1['ds']]

X = np.array([Month, Year]).T

Y_month = [6,7,8,9,10,11,12]
Y_year = [2022,2022,2022,2022,2022,2022,2022]

x = pd.date_range(start = '2022-06-01', end = '2023-01-01', freq = 'm')
x = x.to_pydatetime()

Y = np.array([Y_month, Y_year,Y_index]).T

meses = ['Junho','Julho','Agosto','Setembro','Outubro','Novembro','Dezembro']
meses = pd.DataFrame(meses, columns=['Meses'])

data_df = pd.DataFrame(Y)
data_df = data_df.set_index(2)

Y = np.array([Y_month, Y_year]).T

# fit the model
my_rf = RandomForestRegressor(n_estimators = 400)
my_rf.fit(X, df1.y.values)
 
# predict on the same period
preds = my_rf.predict(X)
preds1 = my_rf.predict(Y)
 
# plot what has been learned

st.subheader("Previsão com o modelo 1")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df1.ds,
                         y=df1.y.values,
                         name='Valores reais',
                         line_color='black', mode='lines+markers'))

fig4.add_trace(go.Scatter(x=df1.ds,
                         y=preds,
                         name='Valores previsto',
                         line_color='green', mode='lines+markers'))

fig4.add_trace(go.Scatter(x=x,
                         y=preds1,
                         name='Valores previsto',
                         line_color='red', mode='lines+markers'))

st.plotly_chart(fig4)

arma_rmse = np.sqrt(mean_squared_error(df1['y'],preds))
st.subheader('Erro: ') 
st.subheader(arma_rmse) 
 
preds1 = pd.DataFrame(preds1, columns=['Valor'])
preds1['Meses'] = meses['Meses'] 
preds1 = preds1[['Meses','Valor']]
preds1

# fit the model
my_xgb = xgb.XGBRegressor(objective= 'reg:squarederror',
                          learning_rate = 0.1,
                          n_estimators =100, 
                          max_depth = 3, 
                          seed = 0)

my_xgb.fit(X, df1.y.values)
 
# predict on the same period
preds2 = my_xgb.predict(X)
preds3 = my_xgb.predict(Y)

# plot what has been learned
st.subheader("Previsão com o modelo 2")

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df1.ds,
                         y=df1.y.values,
                         name='Valores reais',
                         line_color='black', mode='lines+markers'))

fig5.add_trace(go.Scatter(x=df1.ds,
                         y=preds2,
                         name='Valores previsto',
                        line_color='green', mode='lines+markers'))

fig5.add_trace(go.Scatter(x=x,
                         y=preds3,
                         name='Valores previsto',
                         line_color='red', mode='lines+markers'))

st.plotly_chart(fig5)
arma_rmse = np.sqrt(mean_squared_error(df1['y'],preds2))
st.subheader('Erro: ') 
st.subheader(arma_rmse) 

preds3 = pd.DataFrame(preds3, columns=['Valor'])
preds3['Meses'] = meses['Meses'] 
preds3 = preds1[['Meses','Valor']]
preds3
