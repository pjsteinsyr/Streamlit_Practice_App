import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit-learn as sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title("Peter's Practice Site")
data = pd.read_csv("data.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

st.write(x)
st.write(y)
