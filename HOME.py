import streamlit as st
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

# Page title
st.set_page_config(page_title='NWDAF',
                   layout="wide")
st.title('NWDAF IMPLEMENTATION')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app shows the use of pre-trained ML algoritms to replicate the capabilities of NWDAF.')
  st.markdown('**How to use the app?**')
  st.warning('To engage with the app, 1. Go to the desired NWDAF functionality 2. Upload yout data and analyse it using the UI.')
  
st.subheader(" ")
st.image('data/home.PNG', use_column_width='auto')

with st.sidebar:
    st.title('NWDAF DASHBOARD')
