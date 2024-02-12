import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("Peter's Practice Site")
data = pd.read_csv("data.csv")
st.write(data.head())
fig, ax = plt.subplots()
ax.hist(data.Height, bins = 10)
st.pyplot(fig)
