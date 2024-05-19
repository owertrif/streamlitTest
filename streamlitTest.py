import streamlit as st
import numpy as np
import pandas as pd

st.text_input('Your name', key = 'name')
st.session_state.name