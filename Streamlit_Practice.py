import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

st.title("Peter's Practice Site")
data = pd.read_csv("data.csv")
fig, ax = plt.subplots()
ax.hist(data.height, bins = 10)
st.pyplot(fig)
