import streamlit as st
import pandas as pd
st.write('Here is my first attempt to visualise table: ')
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df