import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

st.title("Peter's Test Site")
st.header("**This site is a test for IST 300**")
st.subheader("This site will display some of the basic functionalities of Streamlit.")
st.caption("This is what a caption looks like.")
st.code("""x = 'This is what code looks like in Streamlit'
print(x)""")
st.latex(r''' a+a r^1+a r^2+a r^3 ''')
st.image("SyracuseUniversity.jpg")
st.checkbox('yes')
st.button('Click')
st.radio('Pick your gender',['Male','Female'])
st.selectbox('Pick your gender',['Male','Female'])
st.multiselect('choose a planet',['Jupiter', 'Mars', 'neptune'])
st.select_slider('Pick a mark', ['Bad', 'Good', 'Excellent'])
st.slider('Pick a number', 0,50)
st.number_input('Pick a number', 0,10)
st.text_input('Email address')
st.date_input('Travelling date')
st.time_input('School time')
st.text_area('Description')
st.file_uploader('Upload a photo')
st.color_picker('Choose your favorite color')
st.balloons()
st.progress(10)
with st.spinner('Wait for it...'):
    time.sleep(10)
df= pd.DataFrame(np.random.randn(10, 2),
                 columns=['x', 'y'])
st.line_chart(df)
