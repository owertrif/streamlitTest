import streamlit as st
import numpy as np
import pandas as pd

add_selectbox = st.sidebar.selectbox(
  'How would you like to be contacted? ',
  ('Email', 'Phone', 'Home phone')
)

add_slider = st.sidebar.slider(
  'Select a range of values',
  0.0,100.0,(25.0,75.0)
)