import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Peter's Practice Site")
data = pd.read_csv("data.csv")
st.header('The Data')
st.write(data.head())


x = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
train = pd.merge(x_train, y_train,  left_index=True, right_index=True)

st.header('The Training Data')
st.write(train.head())
fig = px.scatter(train, x = 'Height', y = 'Weight') 
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

st.header("Linear Regression")
st.caption("A linear regression model will now be fit to the training data.")

fig1 = px.scatter(train, x = 'Height', y = 'Weight') 

st.plotly_chart(fig1, theme="streamlit", use_container_width=True)

example = train
example['pred'] = lin_reg.predict(x_train)
st.write(example)
#fig2 = px.line(example, x = "Height", y = "pred")
#st.plotly_chart(fig2, theme="streamlit", use_container_width=True)
lin_pred = lin_reg.predict(x_test)
results = pd.DataFrame(data = {"pred": lin_pred, "act": y_test})
results['hgt'] = x_test
st.write(results.head())

