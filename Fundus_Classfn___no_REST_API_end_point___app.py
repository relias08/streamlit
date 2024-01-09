# Entire code worked perfectly fine !

# =======================================================================================
# *** 1st Tier ***
# This file .
# - This file should basically be on a separate computer ie. 1st Tier (or Web Server) and should have contained only the code for creating the Streamlit web page 
# ie. 1st Tier stuff, but I have also built the best model within this file itself instead of sending a requests.post() to a Rest API end point presented by mlflow running 
# on a separate 2nd Tier computer
- This file should have been run using a terminal of the 1st Tier computer using the foll. command:
# !streamlit run some_app.py. This will result in the web page being available by default on port 8501 of the 1st Tier. 
# - We would then use a browser on a Client computer to send a HTTP request to the 1st Tier computer as follows --- "https://ip_add_of_1st_Tier_computer:8501"

# *** 2nd Tier ***
# Current file with Streamlit code should actually send a request.post() to a REST API end-point on a separate computer ie. 2nd Tier (or Application Server). We should 
# have mlflow running on the 2nd Tier computer by running the foll. command on a terminal of the 2nd Tier computer ---  "!mlflow --sqlite ...."
# Remember that Mlflow running on the 2nd Tier computer should have the best model logged in the 'Model' section of the Mlflow UI

# *** 3rd Tier ***
# This will be a separate computer on which some database will be running
# =======================================================================================

# This is how I am currently running stuff --- Colab is my 1st and 2nd Tier combined
# - We call the current file from inside the following file running on Colab:
# [Deploy Fundus model using Streamlit --- Colab + ngrok.ipynb](https://colab.research.google.com/drive/1W2VmuhsIKEwiAJQoSc0kaZKmOk96DSV8#scrollTo=8UXnKWbBOtGQ)
# using the following command --- "!streamlit run https://github.com/relias08/streamlit/blob/main/Fundus_Classfn___no_REST_API_end_point___app.py&>/dev/null&"

# - Current file (which contains code for creating the Streamlit web page) is used to launch the Streamlit web page by calling it inside the following file running on Colab:
# (so Colab is like the 2nd Tier I guess): 
#  
# which has the following line of code --- "!streamlit run https://github.com/relias08/streamlit/blob/main/Fundus_Classfn___no_REST_API_end_point___app.py&>/dev/null&"

# IMPORTANT NOTE --- in this file we are running predictions using model(input) and not requests.post() ie. not that RestAPI end-point from Mlflow !

# This file is based on --- "jcharis___mitochondria___app.py"

# Important Points:
# - st.image() which displays an image on the Streamlit web page requires PIL objects !!!
# - never use .astype('int') --- always use .astype(np.uint8)   # found out the hard way!   Big Q - what is the difference ???


# ---------------------------------------------------------------------------------
#****** All Pytorch imports for Computer Vision ******

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

import os
from pathlib import Path
from tqdm import tqdm
import PIL

import torchvision
from torchvision.datasets import MNIST
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, Subset, WeightedRandomSampler
import torchmetrics

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

# ---------------------------------------------------------------------------------
#****** Import the model and feature extractor from HuggingFace Transformers library ******

from transformers import ViTFeatureExtractor, ViTForImageClassification

model_name = 'google/vit-base-patch16-224-in21k'
num_classes = 8

id2label = {0:'Normal', 1:'Diabetes', 2:'Glaucoma', 3:'Cataract', 4:'Age related Macular Degeneration', 5:'Hypertension', 6:'Pathological Myopia', 7:'Other diseases/abnormalities'}
label2id = {v:k for k,v in id2label.items()}

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name,
                                num_labels = num_classes,         # num_classes = 8
                                label2id = label2id,
                                id2label = id2label)  #.to(device)   

# Ritheesh did not use .to(device) --- this is applied inside Trainer automatically!

# ---------------------------------------------------------------------------------
#****** Build the Pytorch Lightning model ******
# Ayayooo - I don't think its necessary to build the model using Pytorch Lightning since we are not training the model 
# in this lesson. We are just taking the ViT model from Transformers Library, loading the pre-trained wts onto it and running predictions!

# # Nice if I could put this block into a separate file and call it here using for eg. --- 'from model import Classifier'

from torchmetrics import Accuracy

class Classifier(pl.LightningModule):
    def __init__(self, model, lr: float = 2e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        self.forward = self.model.forward
        self.val_acc = Accuracy(
            task='multiclass' if model.config.num_labels > 2 else 'binary',
            num_classes=model.config.num_labels
        )

    def training_step(self, batch, batch_idx):     # simply create code for forward pass & loss in this method !!!
        outputs = self(**batch)    # Aladdin's vid #2 (5:06) says we can do --- self.forward(**batch)
        self.log(f"train_loss", outputs.loss)
        return outputs.loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        self.log(f"val_loss", outputs.loss)
        acc = self.val_acc(outputs.logits.argmax(1), batch['labels'])
        self.log(f"val_acc", acc, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):    # see Aladdin's vid #2 (7:55)
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)    # Q - can we include scheduler in here ?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
best_model = Classifier(model, lr=2e-5).to(device)       # so best_model is a Pytorch Lightning model !

#****** LOAD PRE-TRAINED WEIGHTS ON TO ABOVE PYTORCH LIGHTNING MODEL ******
# (this model can be used only to run INFERENCE using the trained model ie. not for continued training from saved checkpoint)

checkpoint_path = "/content/gdrive/MyDrive/Colab Notebooks/_CNN___Main/___1___ViT/tb_logs123/Test___June_19/version_3/checkpoints/epoch=15-step=880.ckpt"
checkpoint = torch.load(checkpoint_path)   
best_model.load_state_dict(checkpoint['state_dict'])

# ---------------------------------------------------------------------------------
# ****** STREAMLIT STUFF ******

def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":       # Note --- 'Home' is for images
        st.subheader("Classification of Retinal Fundoscopy Images")

        # Upload raw input image on Streamlit web site, then pass it to PIL.Image.open() - see below!:
        image_file = st.file_uploader("Upload Input Image", type=['png','jpeg','jpg'])         
        
        if image_file is not None:
            # file_details = {"Filename":image_file.name, "FileType":image_file.type, "FileSize":image_file.size}
            # st.write(file_details)

            # Import input image into a PIL object
            img = PIL.Image.open(image_file)         # jpg img => PIL Object --- img.size -> (1024, 768)
            img = img.resize((256, 256))      

            # -----------------------------
            # For displaying input image by itself in Streamlit
            img_ = img.copy()     # Note - img_ is a PIL Obj & st.image() below requires a PIL obj to display it on the Streamlit web page !!!
            st.write('Input image')
            st.image(img_, width=250)

            st.write('Ground Truth: ')          

            # ****** MAKE PREDICTION ******
            # First pre-process the input image using feature_extractor (ie. normalize + re-size):
            processed_image = feature_extractor(img_, return_tensors = 'pt')  # processed_image is a dict with 1 key ie. 'pixel_values'
            final_image = processed_image['pixel_values'][0].unsqueeze(0)    # final_image.shape ---> torch.Size([1, 3, 224, 224])
            final_image = final_image.to(device)

            # Run prediction on trained ViT model:      
            best_model.eval()
            with torch.no_grad():
              output = best_model(final_image)   # note that 'output' is a dictionary with 1 key ie 'logits'

# output ---> ImageClassifierOutput(loss=None, 
#                                   logits=tensor([[-0.6122, -0.7214,  3.5567, -0.4223, -0.0570, -0.4900, -0.1830, -0.6580]], device='cuda:0'), 
#                                   hidden_states=None, attentions=None)

            output_class = output.logits.argmax(dim = 1)
            prediction = id2label[output_class.detach().cpu().item()]     # returns for eg. --- 'Glaucoma'
            st.write(f'Prediction: {prediction}')

            # Ground Truth: --- need to do this

    else:    # this is for the 'About' tab on the Streamlit web page I guess
      st.subheader("About")
      st.info("Built with Streamlit")
      st.info("Jesus Saves")


if __name__ == '__main__':
    main()
