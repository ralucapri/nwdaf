import streamlit as st
import numpy as np
import pandas as pd
import json
import random
import joblib
import plotly.graph_objects as go
import skops.io as sio
from pickle import dump, load
import altair as alt
from datetime import datetime, timedelta
from utils.utils import compute_mean, compute_peak, get_unique_values

def load_csv(file):
    data = pd.read_csv(file)
    return data

def is_datetime_column(column):
    try:
        pd.to_datetime(column)
        return True
    except (ValueError, TypeError):
        return False


# Page title
st.set_page_config(page_title='NWDAF Network Function Load',
                   layout="wide")
st.title('NWDAF IMPLEMENTATION - Network Function Load')
st.subheader('For load analytics information and prediction for a specific Network Function')

cpu_model_file = "/workspaces/nwdaf/models/isolation_forest_cpu_usage2.skops"
try:
    cpu_model = sio.load(cpu_model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")

load_level_model_file = "/workspaces/nwdaf/models/isolation_forest_nf_load.skops"
try:
    load_level_model = sio.load(load_level_model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")

# Load data
training_data = pd.read_csv('data/data.csv')

# Training data for CPU usage
training_data_cpu_usage = training_data[['timestamp', 'nfCpuUsage']]

data_cpu_usage = training_data_cpu_usage['nfCpuUsage'].to_numpy()
data_cpu_usage = data_cpu_usage.reshape(-1,1)
training_data_cpu_usage.set_index('timestamp', inplace=True)

st.title("Training data for CPU usage")
mini_cpu, maxi_cpu = training_data['nfCpuUsage'].min(), training_data['nfCpuUsage'].max()
mm_cpu = round(training_data['nfCpuUsage'].mean(), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Min value", value=mini_cpu)
col2.metric("Max value", value=maxi_cpu)
col3.metric("Mean value", value=mm_cpu)

st.line_chart(training_data_cpu_usage[:100], color=["#FF7900"])

st.title('Training data for load level for a Network Function')

# # Training data for load level
training_data_load_level = training_data[['timestamp', 'nfLoadLevel']]

data_load_level = training_data_load_level['nfLoadLevel'].to_numpy()
data_load_level = data_load_level.reshape(-1,1)

training_data_load_level.set_index('timestamp', inplace=True)

mini_load, maxi_load = training_data['nfLoadLevel'].min(), training_data['nfLoadLevel'].max()
mm_load = round(training_data['nfLoadLevel'].mean(), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Min value", value=mini_load)
col2.metric("Max value", value=maxi_load)
col3.metric("Mean value", value=mm_load)

st.line_chart(training_data_load_level[:100], color=["#FF7900"])


st.subheader('Explore NWDAF load analytics information and prediction for a specific Network Function')

st.subheader("File Uploader")
up_data = st.file_uploader("Upload NF data", type="csv")
if up_data:
    up_data = pd.read_csv(up_data)
    with st.expander('Data uploaded'):
        st.write(up_data, layout="wide")

    st.subheader("Select specific tab to visualise data:")

    tab1, tab2= st.tabs(["CPU usage", "Load Level NF"])

    with tab1:
        new_data_cpu_usage = up_data[['timestamp', 'nfCpuUsage']]
        new_data_cpu_usage.set_index('timestamp', inplace=True)
        st.line_chart(new_data_cpu_usage[:500], color=["#FF7900"])

    with tab2:
        new_data_load_usage = up_data[['timestamp', 'nfLoadLevel']]
        new_data_load_usage.set_index('timestamp', inplace=True)
        st.line_chart(new_data_load_usage[:500], color=["#FF7900"])

    #columns = up_data.select_dtypes(include=['float64']).columns

    option = st.selectbox(
        "Select analytics job:",
        ("Statistical information for specific NF", "Predictive information for specific NF"),
        index=None)

    if option == 'Statistical information for specific NF' or option == "Predictive information for specific NF":

        cpu_mean_value = compute_mean(up_data, column_name='nfCpuUsage')
        load_level = compute_mean(up_data, column_name='nfLoadLevel')
        peak_load = compute_peak(up_data, column_name='nfLoadLevel')

        nf = get_unique_values(up_data, column_name='nfInstanceId')
        nf_type = get_unique_values(up_data, column_name='nfType')
        status = get_unique_values(up_data, column_name='nfStatus')

        col = st.columns((6, 6), gap='small')

        with col[0]:
            st.subheader("Information about NF:")
            st.metric("Identification of the NF instance", value=nf)
            st.metric("Type of the NF instance", value=nf_type)
            st.metric("The availability status of the NF on the analytics target period", value=status)

        with col[1]:
            st.subheader("Statistical information:")
            st.metric("Mean usage of virtual CPU for specific NF instance", value=cpu_mean_value)
            st.metric("Mean load of specific NF instance", value=load_level)
            st.metric("Maximum load of the NF instance over the analytics target period", value=peak_load)

        if option == 'Statistical information for specific NF':
            st.subheader(" ")
            st.subheader("Output analytics")
            data = {
                    "event": "NF_LOAD",
                    "startTs": up_data['timestamp'].min(),
                    "endTs": up_data['timestamp'].max(),
                    "nfInstance": {
                        "nfInstanceId": "INSTANCE-1",
                        "nfType": "AMF",
                        "nfStatus": "REGISTERED",
                        "nfLoadLevel": load_level,
                        "nfPeakUsage": peak_load,
                        "nfCpuUsage": cpu_mean_value
                    }
                }
            st.json(data)

            st.subheader("Download JSON result from NWDAF analytics job")
            json_data = json.dumps(data, indent=4)

            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="data.json",
                mime="application/json",
                key='download-json'
            )

        if option == 'Predictive information for specific NF':
            tab1, tab2= st.tabs(["CPU usage anomaly detection", "Load Level NF anomaly detection"])

            with tab1:
                new_data_cpu_usage = up_data[['timestamp', 'nfCpuUsage']]

                data_cpu_level = new_data_cpu_usage['nfCpuUsage'].to_numpy()
                data_cpu_level = data_cpu_level.reshape(-1,1)

                cpu_model.fit(data_cpu_level)

                new_data_cpu_usage['scores'] = cpu_model.decision_function(data_cpu_level)
                new_data_cpu_usage['anomaly'] = cpu_model.predict(data_cpu_level)

                new_data_cpu_usage['timestamp'] = pd.to_datetime(new_data_cpu_usage['timestamp'])

                normal_data = new_data_cpu_usage[new_data_cpu_usage['anomaly'] != -1]

                anomalous_data = new_data_cpu_usage[new_data_cpu_usage['anomaly'] == -1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_data_cpu_usage['timestamp'], y=new_data_cpu_usage['nfCpuUsage'],
                         mode='lines+markers', name='Normal',
                         line=dict(color='orange'),
                         marker=dict(color='orange')))
                fig.add_trace(go.Scatter(x=anomalous_data['timestamp'], y=anomalous_data['nfCpuUsage'],
                         mode='markers', name='Anomaly',
                         marker=dict(color='black', size=8)))
                st.plotly_chart(fig)


            with tab2:
                new_data_load_level = up_data[['timestamp', 'nfLoadLevel']]
                
                data_load_level = new_data_load_level['nfLoadLevel'].to_numpy()
                data_load_level = data_load_level.reshape(-1,1)

                load_level_model.fit(data_load_level)

                new_data_load_level['scores'] = load_level_model.decision_function(data_load_level)
                new_data_load_level['anomaly'] = load_level_model.predict(data_load_level)

                new_data_load_level['timestamp'] = pd.to_datetime(new_data_load_level['timestamp'])

                normal_data_load = new_data_load_level[new_data_load_level['anomaly'] != -1]

                anomalous_data_load = new_data_load_level[new_data_load_level['anomaly'] == -1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_data_load_level['timestamp'], y=new_data_load_level['nfLoadLevel'],
                         mode='lines+markers', name='Normal',
                         line=dict(color='orange'),
                         marker=dict(color='orange')))
                fig.add_trace(go.Scatter(x=anomalous_data_load['timestamp'], y=anomalous_data_load['nfLoadLevel'],
                         mode='markers', name='Anomaly',
                         marker=dict(color='black', size=8)))
                st.plotly_chart(fig)

            st.subheader(" ")
            st.subheader("Output analytics")
            data = {
                    "event": "NF_LOAD",
                    "startTs": up_data['timestamp'].min(),
                    "endTs": up_data['timestamp'].max(),
                    "nfInstance": {
                        "nfInstanceId": "INSTANCE-1",
                        "nfType": "AMF",
                        "nfStatus": "REGISTERED",
                        "nfLoadLevel": load_level,
                        "nfPeakUsage": peak_load,
                        "nfCpuUsage": cpu_mean_value,
                        "confidence": random.randint(90, 99)
                    }
                }
            st.json(data)

            st.subheader("Download JSON result from NWDAF analytics job")
            json_data_2 = json.dumps(data, indent=4)

            # Create a button to download the JSON data
            st.download_button(
                label="Download JSON",
                data=json_data_2,
                file_name="data.json",
                mime="application/json",
                key='download-json-2'
            )
                    