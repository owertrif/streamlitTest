import streamlit as st
import numpy as np
import pandas as pd

left_column, right_column = st.columns(2)

left_column.button('Click me!')

with right_column:
  chosen = st.radio('Which one would you like to select?', ('1', '2', '3'))