import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from sklearn import metrics

# Title, cover image, and load in data.
st.title("Peter's Practice Site")
st.image('weight.jpg')
data = pd.read_csv("data.csv")

# Section that displays the data.
st.header('The Data')
st.write(data.head())

# Create training and testing datasets.
x = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
train = pd.merge(x_train, y_train,  left_index=True, right_index=True)

# Display training data and a plot of the data.
st.header('The Training Data')
st.write(train.head())
fig = px.scatter(train, x = 'Height', y = 'Weight') 
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# Define LinearRegression and fit training data to the model.
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

# Section to display the linear regression and show it's fit.
st.header("Linear Regression")
st.caption("A linear regression model will now be fit to the training data.")
example = train
example['pred'] = lin_reg.predict(x_train)
fig = px.scatter(x=example['Height'], y=example['Weight'], labels = {'x': 'Height', 'y': 'Weight'})
fig.add_trace(go.Scatter(x=example['Height'], y=example['pred'], marker = {'color' : 'red'}, name="Model"))
st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# 
st.header('Model Testing')
lin_pred = lin_reg.predict(x_test)
results = pd.DataFrame(data = {"pred": lin_pred, "act": y_test})
results['hgt'] = x_test
st.write(results)
st.write(f'R square = {metrics.r2_score(y_test, lin_pred)}')
st.write(f'Mean squared Error = {metrics.mean_squared_error(y_test, lin_pred)}')

st.sidebar.header('Try it for yourself!')
number = st.sidebar.number_input('Insert your height (cms)')
ur_number = pd.DataFrame(data = {"Height" : [number]})
if st.sidebar.button("Print Weight"):
    # Display the entered number when the button is clicked
    st.sidebar.write(f"Your estimated weight is {round(lin_reg.predict(ur_number)[0],2)} kgs.")
    st.balloons()
