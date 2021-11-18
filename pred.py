import streamlit as st 
import mysql.connector
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from fbprophet.plot import plot_components_plotly
from  fbprophet.diagnostics import cross_validation
from  fbprophet.diagnostics import performance_metrics
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import webbrowser



#Page expands to full width
st.beta_set_page_config(layout="wide")

from bokeh.models.widgets import Div

if st.button('Back'):
    js = "window.open('http://testingbig.rf.gd/big1/main/salesreport.php?d1=0&d2=0')"  # New tab or window
    js = "window.location.href = ('http://testingbig.rf.gd/big1/main/salesreport.php?d1=0&d2=0')"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)
    

st.title("Predictive Analytics")

#About
expander_bar = st.beta_expander("About")
expander_bar.markdown("""
**Context:** Time-series forecasting using *Prophet* model to ...
""")

product = ("All Product","Big Buko Pie / Box","Mini Buko Pie Box","Mini Buko Pie Piece","Macaroons","Macapuno Balls","Coffee",
           "Buko Juice 1L Bottle","Buko Shake 1L Bottle","Macapuno Shake 1L Bottle","Buko Juice 12oz Cup","Buko Juice 16oz Cup",
           "Buko Juice 22oz Cup","Buko Shake 12oz Cup","Buko Shake 16oz Cup","Buko Shake 22oz Cup","Hot Choco","Macapuno Shake 12oz Cup",
           "Macapuno Shake 16oz Cup","Macapuno Shake 22oz Cup","Buko Juice 350ml Bottle","Buko Shake 350ml Bottle",
           "Buko Shake 500ml Bottle","Macapuno Shake 350ml Bottle","Macapuno Shake 500ml Bottle")




selected_product = st.selectbox("Select product for prediction:",product)


n_days = st.slider('Days of prediction:', 1, 7)

connection = mysql.connector.connect(host = 'sql6.freesqldatabase.com',user = 'sql6451777', passwd = 'RyxluaSUqz', db = 'sql6451777')


if selected_product == "All Product":
  sales = pd.read_sql_query("SELECT * FROM sales_order WHERE date >= '2021-03-01 00:00:00'",  connection)
else:
  sales = pd.read_sql_query("SELECT * FROM sales_order WHERE date >= '2021-03-01 00:00:00' and name = '%s' " % selected_product, connection)



sales_new = sales[['qty','date']]
sales_new['date'] = pd.to_datetime(sales_new['date']).dt.date
sales_new = sales_new.rename(columns = {'qty': 'y', 'date': 'ds'}, inplace = False)
sales_new = sales_new.groupby('ds')['y'].sum()
sales_new = pd.DataFrame(sales_new)
sales_new.reset_index(level=0, inplace=True)


m = Prophet(interval_width=0.88).add_seasonality(name='monthly', period= 30.5,fourier_order=14)
model = m.fit(sales_new)

future = m.make_future_dataframe(periods= n_days,freq='D')
forecast = m.predict(future)

st.title("%s Day/s " % n_days + "Time-Series Forecast for : \n ")
st.title("%s " % selected_product)


#TABLE
zxc = forecast.tail(n_days)
zxc= zxc[['ds','yhat']]
zxc= zxc.rename(columns = {'ds': 'Date', 'yhat': 'Predicted'})
import datetime
zxc['Day'] = zxc['Date'].dt.strftime("%B %d, %Y at %A")
#zxc['Date'] = zxc['Date'].apply(lambda x: str(x)[0:10])

qwe = zxc[['Day', 'Predicted']]
st.write(qwe)

#ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, DayLocator

fig, ax = plt.subplots(figsize=(25, 5))

#Actual
ax.plot(sales_new.ds, sales_new.y,marker='o', markerfacecolor='green',
 markersize=5, color='lightgreen', linewidth=4, label='Actual')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(DateFormatter('%B'))

#For pred_val
asd = forecast.tail(n_days)
asd = asd[['ds','yhat']]
asd = asd.rename(columns = {'ds': 'Date', 'yhat': 'Predicted'})
#Predicted
ax.plot(asd.Date, asd.Predicted,marker='o', markerfacecolor='blue',
 markersize=5, color='skyblue', linewidth=4, label='Predicted')
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(DateFormatter('%B'))

for tick in ax.xaxis.get_major_ticks():
  tick.label.set_fontsize(14)

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.5,
        alpha=0.2)

st.markdown("<h1 style='text-align: center; color: black;'>Time-series Forecast</h1>", unsafe_allow_html=True)


st.plotly_chart(fig, use_container_width=True)
#oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
st.subheader("Explore Time Series Forcast")
labe_title = st.markdown("<p style='font-size:22px'><b>Time Series Analysis with Uncertainty Area</b></p>", unsafe_allow_html=True)
with st.beta_expander(label="Time Series Analysis with Uncertainty Area"):
  st.markdown("<p style='font-size:22px'><b>The chart below shows the actual value together with the predicted value and the area of uncertainty.</b></p>", unsafe_allow_html=True)
  st.markdown("<p style='font-size:22px'><b>Uncertainty area or Confidence band in this are most of the data falls. As a Decision maker points out of the Uncertainty are worth understanding, sure there is something special about it.</b></p>", unsafe_allow_html=True)
  st.markdown("<p style='font-size:22px'><b>THIS WILL BE FOR GOOD OR FOR WORSE.</b></p>", unsafe_allow_html=True)


  plot1 = plot_plotly(m, forecast)
  st.plotly_chart(plot1, use_container_width=True)

#st.title("\n Time-Series Model Assessment")
#b = forecast.replace(forecast[['yhat_lower','yhat_upper','yhat']].head(-n_days).values,np.NaN)
#plot1 = plot_plotly(m, b)
#plot1


#st.title("Time-Series Components")
#plot2  = plot_components_plotly(m, forecast)
#plot2

