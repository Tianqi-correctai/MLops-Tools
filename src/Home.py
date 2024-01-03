import os
from pathlib import Path
from cvat_api import CVATAPI
import streamlit as st
import pandas as pd
import json

# Parameters load from the config file
os.chdir(Path(__file__).resolve().parent)
json_file = open("config.json")
params = json.load(json_file)


# Instantiate the CVAT API
# CVAT = CVATAPI(cvat_url)
# CVAT.authenticate(username, password)

def get_params():
    for key, value in params.items():
        st.session_state[key] = value

def main():
    # Correct-AI MLOps Tools
    st.markdown(open("README.md").read())

main()
