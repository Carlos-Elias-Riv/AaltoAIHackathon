import streamlit as st
import pandas as pd
import numpy as np
from streamlit_javascript import st_javascript


url = st_javascript("await fetch('').then(r => window.parent.location.href)")

if "medicine_id" in st.query_params:
    graphid = st.query_params["medicine_id"]
    st.title(graphid.strip("'"))
    plotInfo = {"Year": [i for i in range(2000, 2020)], "Data": [i for i in range(0, 20)]}  
    st.line_chart(plotInfo,x="Year", y="Data")
    st.link_button("Home", url=str(url).split("/?")[0])

else:
    
    data = pd.read_csv("medicines.csv")
    st.title('Medicine visualization')

    text_search = st.text_input("Search medicines", value="")
    result = data[data["Company"].str.contains(text_search)]

    
    if text_search:
        for row in result["Company"]:
            st.link_button(row, url=url+ "?medicine_id='"+ row + "'")
                
                
                