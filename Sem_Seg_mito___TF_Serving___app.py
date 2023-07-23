# I HAVE NOT YET TESTED THIS FILE --- SO NEED TO CHECK IF IT WORKS PROPERLY !!!

# For how to use this file, see following super file on Laptop - "204__how_to_run_predictions_in_Streamlit" --- stored in 
# following folder on Laptop --- "__Deploying_Models___MAIN/___Streamlit_and_mlflow/worked___mitichondria___Sreeni_204"

# Basically if we are deploying the Sem_Seg_mito model using Streamlit from Colab using ngrok, this file should be called inside the following Colab file in My Google Drive:
# [Deploy Sem Seg mito model using Streamlit___Colab+ngrok___no_TFServing___worked](https://drive.google.com/drive/folders/141zJ3AU4KA4rM2j5xA4ljd1UVWgQbOS-)
# as follows --- !streamlit run https://github.com/relias08/streamlit/edit/main/Sem_Seg_mito___TF_Serving___app.py&>/dev/null&

# Consider the Github version of this file as original ie. 
# [Sem_Seg_mito___TF_Serving___app.py](https://github.com/relias08/streamlit/new/main)

# In this file we are running predictions on the Mitochondrial Sem Seg model by using requests.post() to send data to 
# that RestAPI end-point from TF Serving (instead of using model.predict)! 

# This file was created by adapting the following file in Github --- [tfserving___bare_min_version_worked.py](https://github.com/relias08/streamlit/blob/main/tfserving___bare_min_version_worked.py)

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

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":       # Note --- 'Home' is for images
        st.subheader("Sem Seg of Mitochondria in Electron Microscopy Images")

        # Upload input image on Streamlit web site:
        image_file = st.file_uploader("Upload Input Image", type=['png','jpeg','jpg'])      # this image_file is ready to go into PIL.Image.open() !

        # Upload mask/target/groud_truth image on Streamlit web site:
        mask_file = st.file_uploader("Upload Mask Image ie. ground truth", type=['png','jpeg','jpg'])        # this mask_file is ready to go into PIL.Image.open() !
        
        
        if image_file is not None and mask_file is not None:
            # file_details = {"Filename":image_file.name, "FileType":image_file.type, "FileSize":image_file.size}
            # st.write(file_details)

            # Import input image into a PIL object
            img = Image.open(image_file)         # img is now a Pillow object, with img.size -> (1024, 768)
            img = img.resize((256, 256))      

            # -----------------------------
            # For displaying input image alone in Streamlit
            img_ = img.copy()     # Note --- st.image() below which displays the image on the Streamlit web page requires a PIL object !!!
            # st.write('Input image')
            # st.image(img_, width=250)

            # -----------------------------
            # For displaying the mask image alone in Streamlit (ie. ground truth in this case): 
            mask = Image.open(mask_file)      # converts 'mask_file' to a PIL object
            mask = mask.resize((256, 256))    # just to make consistent with input image & pred image
            # st.image(mask, width=250)
            # -----------------------------

            # ****** MAKE PREDICTION ****** :
            # Convert PIL object to np array, then pre-process the image ie. normalize + expand_dims
            img = np.array(img)     # convert PIL image to np array
            arr = np.expand_dims(np.expand_dims(normalize(img), 2), 0) # normalize + expand_dims; img.shape -> (1, 256, 256, 1)
         
            data = json.dumps({"signature_name": "serving_default", 
                               "instances": arr.tolist()})

            # Next line is the RestAPI end-point created by TF Serving
            DEPLOYED_ENDPOINT = "http://localhost:8601/v1/models/mitochondria_model:predict"              

            headers = {"content-type": "application/json"}     # works even without headers --- need to verify!
            r = requests.post(url = DEPLOYED_ENDPOINT, data=data, headers=headers)
            st.write(r)
            
            pred = np.array(r.json()['predictions'])     # pred.shape ---> (1, 256, 256, 1)
            
            threshold = .3
            # pred = (pred[0, :, :, 0] > threshold).astype('int')*255 # dont use 'int'
            pred = (pred[0, :, :, 0] > threshold).astype(np.uint8)*255    # pred[0, :, :, 0].shape ---> (256, 256)    
    
            # For displaying the predicted image alone:
            pred_ = Image.fromarray(pred)     # np array => PIL Object
            # st.write('prediction')
            # st.image(pred_, width = 250)

            # For displaying the input image, ground truth & prediction side by side - remember that st.image() requires 
            # PIL objects !!!:
            #images = [img_, mask, pred_]
            #st.image(images, caption=['input image', 'ground_truth', 'prediction'], width=200)
            
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.image(img_, caption=['input image'])  #'test_img.png',width=360,use_column_width='never')
            with col2:
                st.image(mask, caption=['ground truth']) #'test_img.png',width=360,use_column_width='never')
            with col3:
                st.image(pred_, caption=['prediction'])  #'test_img.png',width=360,use_column_width='never')

    else:
        st.subheader("About")
        st.info("Built with Streamlit")
        st.info("Jesus Saves")



if __name__ == '__main__':
    main()
