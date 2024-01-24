import os
from pathlib import Path
from cvat_api import CVATAPI
import streamlit as st
import pandas as pd
import json
import re
import base64


def markdown_images(markdown):
    # example image markdown:
    # ![Test image](images/test.png "Alternate text")
    images = re.findall(r'(!\[(?P<image_title>[^\]]+)\]\((?P<image_path>[^\)"\s]+)\s*([^\)]*)\))', markdown)
    return images


def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def img_to_html(img_path, img_alt):
    img_format = img_path.split(".")[-1]
    img_html = f'<img src="data:image/{img_format.lower()};base64,{img_to_bytes(img_path)}" alt="{img_alt}" style="max-width: 100%;">'

    return img_html


def markdown_insert_images(markdown):
    images = markdown_images(markdown)

    for image in images:
        image_markdown = image[0]
        image_alt = image[1]
        image_path = image[2]
        if os.path.exists(image_path):
            markdown = markdown.replace(image_markdown, img_to_html(image_path, image_alt))
    return markdown

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
    # st.markdown(open("README.md").read())

    with open("README.md", "r") as readme_file:
        readme = readme_file.read()

        readme = markdown_insert_images(readme)

        with st.container():
            st.markdown(readme, unsafe_allow_html=True)

main()
