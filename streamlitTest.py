import streamlit as st
import numpy as np
import pandas as pd

df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [5, 6, 7, 8]
})

option = st.selectbox(
  'Which column do you want?',
  df['first column']
)

'You chose: ',option