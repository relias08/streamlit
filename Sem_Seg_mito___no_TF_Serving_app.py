# IMPORTANT NOTE --- This file is for running predictions using model.predict() and 
# not requests.post() and that RestAPI end-point from TF Serving!

# Location of this file in Github is:
# https://github.com/relias08/streamlit/blob/main/Semantic_Segmentation_mitochondria___app.py"

# This file is based on --- "jcharis___mitochondria___app.py"

# Important Points:
# - st.image() which displays an image on the Streamlit web page requires PIL objects !!!
# - never use .astype('int') --- always use .astype(np.uint8)   # found out the hard way!   Big Q - what is the difference ???

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import requests
import streamlit as st
import streamlit.components.v1 as stc

from PIL import Image
import tensorflow
from tensorflow.keras.utils import normalize

# Create the model by using the path to saved model on Colab VM on to next line
model = tensorflow.keras.models.load_model("/content/1")

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":       # Note --- 'Home' is for images
        st.subheader("Semantic Segmentation of Mitochondria in Electron Microscopy Images")

        # Upload raw input image on Streamlit web site, then pass it to PIL.Image.open() - see below!:
        image_file = st.file_uploader("Upload Input Image", type=['png','jpeg','jpg'])

        # Upload raw mask/target/groud_truth image on Streamlit web site, then pass it to PIL.Image.open() - see below!::
        mask_file = st.file_uploader("Upload Mask Image ie. ground truth", type=['png','jpeg','jpg'])
        
        
        if image_file is not None and mask_file is not None:
            # file_details = {"Filename":image_file.name, "FileType":image_file.type, "FileSize":image_file.size}
            # st.write(file_details)

            # Import input image into a PIL object
            img = Image.open(image_file)         # jpg img => PIL Object --- img.size -> (1024, 768)
            img = img.resize((256, 256))      

            # -----------------------------
            # For displaying input image by itself in Streamlit
            img_ = img.copy()     # Note - img_ is a PIL Obj & st.image() below requires a PIL obj to display it on the Streamlit web page !!!
            # st.write('Input image')
            # st.image(img_, width=250)

            # -----------------------------
            # For displaying the mask image by itself in Streamlit (ie. ground truth in this case): 
            mask = Image.open(mask_file)      # jpg img => PIL Object
            mask = mask.resize((256, 256))    # just to make consistent with input image & pred image
            # st.image(mask, width=250)
            # -----------------------------

            # ****** MAKE PREDICTION ****** :
            # First convert PIL object to np array & pre-process ie. normalize + expand_dims
            img = np.array(img)      # PIL image => np array
            arr = np.expand_dims(np.expand_dims(normalize(img), 2), 0) # normalize + expand_dims - arr.shape -> (1, 256, 256, 1)
            
            threshold = .3
            pred = (model.predict(arr)[0, :, :, 0] > threshold).astype(np.uint8)*255   
            # pred is np arr of size (256, 256). Pixel values are 0 (black) or 255 (white)

            # -----------------------------
            # For displaying the predicted image by itself:
            pred_ = Image.fromarray(pred)     # np array => PIL Object
            # pred_ = np.array(pred_)
            # st.write('prediction')
            # st.image(pred_, width = 250)
            # -----------------------------

            # For displaying side by side the input image, ground truth & prediction (remember that st.image() requires PIL objects):
            #images = [img_, mask, pred_]
            #st.image(images, caption=['input image', 'ground_truth', 'prediction'], width=200)
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.image(img_, caption=['input image'])  #'test_img.png',width=360,use_column_width='never')
            with col2:
                st.image(mask, caption=['ground truth']) #'test_img.png',width=360,use_column_width='never')
            with col3:
                st.image(pred_, caption=['prediction'])  #'test_img.png',width=360,use_column_width='never')

    else:    # this is for the 'About' tab on the Streamlit web page I guess
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("Jesus Saves")


if __name__ == '__main__':
    main()
