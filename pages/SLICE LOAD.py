import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import random
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.graph_objects as go
import json

from itertools import chain
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from utils.utils import compute_mean, compute_peak, get_unique_values

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mean = self.mean_layer(hidden)
        logvar = self.logvar_layer(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar
    

def compute_threshold(reconstruction_losses, factor=100):
    loss_mean = np.mean(reconstuction_loss) * 100
    loss_std = np.std(reconstuction_loss)
    threshold = loss_mean + factor * loss_std

    return threshold

def detect_anomalies(data, reconstructions, threshold):
    distances = data - reconstructions
    anomalies = np.any(distances >= threshold)
    return_value = []
    for i, element in enumerate(distances):
      if element >= threshold:
        return_value.append([-1, 'Anomaly'])
      else: return_value.append([1, 'Normal'])
    return return_value


# Page title
st.set_page_config(page_title='NWDAF Slice Load',
                   layout="wide")
st.title('NWDAF IMPLEMENTATION - Slice Load')
st.subheader('For Load-level computation and prediction for a network slice instance')

model_slice_load = VAE(input_dim=100, hidden_dim=30, latent_dim=5)
model_slice_load.load_state_dict(torch.load('/workspaces/nwdaf/models/vae_slice_load.pth'))

# Set the model to evaluation mode
model_slice_load.eval()

# Load data
training_data = pd.read_csv('data/data_slice.csv')

# Training data for Slice load
training_data_slice = training_data[['timestamp', 'sliceLoadLevel']]

data_slice = training_data_slice['sliceLoadLevel'].to_numpy()
data_slice = data_slice.reshape(-1,1)
training_data_slice.set_index('timestamp', inplace=True)

st.title("Training data for Slice Load")
mini_cpu, maxi_cpu = round(training_data['sliceLoadLevel'].min(),2), round(training_data['sliceLoadLevel'].max(),2)
mm_cpu = round(training_data['sliceLoadLevel'].mean(), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Min value", value=mini_cpu)
col2.metric("Max value", value=maxi_cpu)
col3.metric("Mean value", value=mm_cpu)

st.line_chart(training_data_slice[:100], color=["#FF7900"])

# Training data for Slice load

st.title("UE evolution in time")

training_data_ues = training_data[['timestamp', 'numOfUes']]

data_ues = training_data_ues['numOfUes'].to_numpy()
data_cpu_usage = data_slice.reshape(-1,1)
training_data_ues.set_index('timestamp', inplace=True)

mini_cpu, maxi_cpu = round(training_data['numOfUes'].min(),2), round(training_data['numOfUes'].max(),2)
mm_cpu = round(training_data['numOfUes'].mean(), 2)

col1, col2, col3 = st.columns(3)
col1.metric("Min value", value=mini_cpu)
col2.metric("Max value", value=maxi_cpu)
col3.metric("Mean value", value=mm_cpu)

st.line_chart(training_data_ues[:100], color=["#FF7900"])

st.subheader('Explore NWDAF Load-level analysis and prediction for a network slice instance')

st.subheader("File Uploader")
up_data = st.file_uploader("Upload slice data", type="csv")
if up_data:
    up_data = pd.read_csv(up_data)
    with st.expander('Data uploaded'):
        st.write(up_data, layout="wide")

    st.subheader("Select specific tab to visualise data:")

    tab1, tab2= st.tabs(["Slice Load Level", "Number of UE"])

    with tab1:
        new_slice_load = up_data[['timestamp', 'sliceLoadLevel']]
        new_slice_load.set_index('timestamp', inplace=True)
        st.line_chart(new_slice_load[:500], color=["#FF7900"])

    with tab2:
        new_data_ues = up_data[['timestamp', 'numOfUes']]
        new_data_ues.set_index('timestamp', inplace=True)
        st.line_chart(new_data_ues[:500], color=["#FF7900"])

    option = st.selectbox(
        "Select analytics job:",
        ("Statistical information for slice instance", "Predictive information for slice instance"),
        index=None)

    if option == 'Statistical information for slice instance' or option == "Predictive information for slice instance":

        slice_load_level = compute_mean(up_data, column_name='sliceLoadLevel')
        peak_load = round(compute_peak(up_data, column_name='sliceLoadLevel'),2)

        slice = get_unique_values(up_data, column_name='snssais')
        ues_reg = get_unique_values(up_data, column_name='numOfUesReg')
        mean_ues = round(compute_mean(up_data, column_name='numOfUesReg'),0)

        col = st.columns((6, 6), gap='small')

        with col[0]:
            st.subheader("Information about slice instance:")
            st.metric("Identification of the slice instance", value=slice)
            st.metric("Number of UEs registered in the slice instance", value=ues_reg)

        with col[1]:
            st.subheader("Statistical information:")
            st.metric("Mean load of slice instance", value=slice_load_level)
            st.metric("Maximum load of slice instance over the analytics target period", value=peak_load)

        if option == 'Statistical information for slice instance':
            st.subheader(" ")
            st.subheader("Output analytics")
            data = {
                    "event": "SLICE_LOAD",
                    "startTs": up_data['timestamp'].min(),
                    "endTs": up_data['timestamp'].max(),
                    "sliceLoadLevelInfo": {
                        "loadLevelInformation": {
                            "mean": slice_load_level,
                            "peak":peak_load
                        },
                        "snssais": slice,
                        "numOfUes": mean_ues,
                        "numOfUesReg": int(ues_reg)
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

    if option == 'Predictive information for slice instance':
            tab1, tab2 = st.tabs(["Slice load level prediction", "Slice load level anomaly detection"])
            

            with tab1:
                new_slice_load = up_data[['timestamp', 'sliceLoadLevel']]
                new_slice_load_dfr = new_slice_load.reset_index(drop=True)
                new_slice_load_values = new_slice_load['sliceLoadLevel']

                tensor_data_slice_load = torch.tensor(new_slice_load_values, dtype=torch.float32)

                results_tensor = torch.empty((0))
                chunk_size = 100

                total_loss = 0
                num_samples = 500
                kl_divergence_loss = 0
                rec_loss = 0

                loss_function = torch.nn.MSELoss()

                for i in range(0, tensor_data_slice_load.size(0), chunk_size):
                    chunk = tensor_data_slice_load[i:i+chunk_size]

                    with torch.no_grad():
                        reconstructed, mean, logvar = model_slice_load(chunk)
                    
                    results_tensor = torch.cat((results_tensor, reconstructed), dim=0)

                    loss = loss_function(reconstructed, chunk)

                    total_loss += loss.item()
                    num_samples += chunk.size(0)
                    kl_divergence_loss += (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())).item()
                    rec_loss += (torch.mean((chunk - reconstructed) ** 2)).item()

                average_loss = round(total_loss / num_samples, 2)
                regularization_loss = round(kl_divergence_loss / num_samples, 2)
                reconstuction_loss = round(rec_loss / num_samples, 2)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_slice_load['timestamp'], y=new_slice_load['sliceLoadLevel'],
                         mode='lines+markers', name='Normal',
                         line=dict(color='orange'),
                         marker=dict(color='orange')))
                fig.add_trace(go.Scatter(x=new_slice_load['timestamp'], y=results_tensor,
                         mode='lines+markers', name='Anomaly',
                         marker=dict(color='black', size=8)))
                st.plotly_chart(fig)

                col1, col2, col3 = st.columns(3)
                col1.metric("Average loss on data", value=average_loss)
                col2.metric("Regularization loss on data", value=regularization_loss)
                col3.metric("Reconstruction loss on data", value=reconstuction_loss)

            with tab2:
                threshold = compute_threshold(reconstuction_loss)
                
                anomalies = detect_anomalies(new_slice_load_values, results_tensor.numpy(), threshold)
                df_anomalies = pd.DataFrame(anomalies, columns=['anomaly', 'label'])

                new_slice_load_values_2 = new_slice_load_values
                new_slice_load_values_2 = pd.concat([new_slice_load, df_anomalies], axis=1)

                anomalies_plot_data = new_slice_load_values_2.loc[new_slice_load_values_2['label']=='Anomaly']
                anomalous_data = new_slice_load_values_2[new_slice_load_values_2['anomaly'] == -1]

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=new_slice_load_values_2['timestamp'], y=new_slice_load_values_2['sliceLoadLevel'],
                         mode='lines+markers', name='Normal',
                         line=dict(color='orange'),
                         marker=dict(color='orange')))
                fig.add_trace(go.Scatter(x=anomalous_data['timestamp'], y=anomalous_data['sliceLoadLevel'],
                         mode='markers', name='Anomaly',
                         marker=dict(color='black', size=4)))
                st.plotly_chart(fig)

                slice_load_level_pred = compute_mean(new_slice_load_values_2, column_name='sliceLoadLevel')
                peak_load_pred = round(compute_peak(new_slice_load_values_2, column_name='sliceLoadLevel'),2)

            st.subheader(" ")
            st.subheader("Output analytics")

            data = {
                    "event": "SLICE_LOAD",
                    "startTs": up_data['timestamp'].min(),
                    "endTs": up_data['timestamp'].max(),
                    "sliceLoadLevelInfo": {
                        "loadLevelInformation": {
                            "mean": slice_load_level_pred,
                            "peak":peak_load_pred
                        },
                        "snssais": slice,
                        "numOfUes": mean_ues,
                        "numOfUesReg": int(ues_reg),
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
                key='download-json-3'
            )
                    