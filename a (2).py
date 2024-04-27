import streamlit as st
import pandas as pd
import numpy as np
from streamlit_javascript import st_javascript
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from collections import defaultdict
import json


url = st_javascript("await fetch('').then(r => window.parent.location.href)")

def reformat_json():
    new_data = defaultdict(dict)
    with open('timeseries.json', 'r') as f:
        data = json.load(f)
        for medicine in data:
            for key in data[medicine]:
                for date in data[medicine][key]:
                    new_data[medicine][date[:10]] = data[medicine][key][date]
    return new_data


def convert_dataframe():
    data = reformat_json()
    for medicine in data:
        databases[medicine] = pd.DataFrame(data[medicine].items(), columns=['date', 'value'])
    return databases

class Availability(nn.Module):
    def __init__(self):
        super(Availability, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x



if "medicine_id" in st.query_params:
    
    
    graphid = st.query_params["medicine_id"]
    graphid = graphid.strip("'")

    


    databases = {}
    databases = convert_dataframe()
    db1 = defaultdict(dict)
    df = pd.read_csv('output.csv')
    for i, row in df.iterrows():
        db1[row['medicine']].update({row['date']: row['score']})

    for key in db1:
        db1[key] = pd.DataFrame(db1[key].items(), columns=['date', 'score'])
    db1 = dict(db1)
    
    try:
        databases[graphid] = pd.merge(databases[graphid], db1[graphid], on='date', how='left', suffixes=('_1', '_2'))
    except Exception:
        pass
    
    databases[graphid]['date'] = pd.to_datetime(databases[graphid]['date'])
    databases[graphid]['year'] = databases[graphid]['date'].dt.year
    databases[graphid]['month'] = databases[graphid]['date'].dt.month
    databases[graphid]['day'] = databases[graphid]['date'].dt.day

    
    databases[graphid].dropna(inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(databases[graphid][['year', 'month', 'day', 'score']].values, databases[graphid]['value'].values, test_size=0.2, shuffle=False)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = Availability()

    model.load_state_dict(torch.load( 'models_new/'+ graphid + ".pth"))
    print('models_new/'+ graphid + ".pth")
    model.eval()

    #with torch.no_grad():
    #    for i in range(len(X_test)):
    #        pred = model(X_test[i, :].reshape(1, 4))



    fig, ax = plt.subplots(figsize=[20,10])
    ax.plot(databases[graphid].loc[:15, 'date'],  y_train.detach().numpy(), label='Train', color='orange')
    ax.plot(databases[graphid].loc[16:, 'date'], model(X_test).detach().numpy(), label='Test', color='blue')
    ax.axvline(x=databases[graphid].loc[15, 'date'], color='red', linestyle='--')

    st.pyplot(fig)

else:
    data = pd.read_csv("medicines.csv")
    st.title('Medicine visualization')

    text_search = st.text_input("Search medicines", value="")
    result = data[data["Medicine"].str.contains(text_search)]

    
    if text_search:
        for row in result["Medicine"]:
            st.link_button(row, url=url+ "?medicine_id='"+ row + "'")
                
                
                