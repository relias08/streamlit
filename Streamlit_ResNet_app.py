mod = 'resnet34'
num_classes = 8

# Entire code worked perfectly fine !

# To launch the Streamlit web page, the current app.py file is called inside the following file stored in Google Drive - 
# [Deploy Fundus model using Streamlit --- Colab + ngrok.ipynb](https://colab.research.google.com/drive/1W2VmuhsIKEwiAJQoSc0kaZKmOk96DSV8#scrollTo=8UXnKWbBOtGQ) 
# using the following line of code:
# !streamlit run https://github.com/relias08/streamlit/blob/main/Fundus_Classfn___no_REST_API_end_point___app.py&>/dev/null&

# IMPORTANT NOTE --- in this file we are running predictions using model(input) and not requests.post() 
# and that RestAPI end-point from Mlflow !

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------------
#****** Import the Resnet model from torch hub ******

cnn = torch.hub.load('pytorch/vision:v0.10.0', mod, pretrained=True)      # 'resnet18', 50
cnn.fc = nn.Linear(cnn.fc.in_features, 8)
cnn = cnn.to(device)    # req because the model was trained on GPU I guess !!!


# Ritheesh did not use .to(device) --- this is applied inside Trainer automatically!

# ---------------------------------------------------------------------------------
#****** Build the Pytorch Lightning model ******
# Ayayooo - I don't think its necessary to build the model using Pytorch Lightning since we are not training the model 
# in this lesson. We are just taking the ViT model from Transformers Library, loading the pre-trained wts onto it and running predictions!

# # Nice if I could put this block into a separate file and call it here using for eg. --- 'from model import Classifier'

from torchmetrics import Accuracy
criterion = nn.CrossEntropyLoss()

class NN(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        # self.model_used = mod    # Q - will this save the mod in mlflow ui? (just like I once saved data used - small_data)
        self.forward = self.model.forward
        self.acc = Accuracy(task='multiclass' if num_classes > 2 else 'binary',
                            num_classes=num_classes)

    def training_step(self, batch, batch_idx):     # simply create code for forward pass & loss in this method !!!
        x_batch, y_batch = batch
        yhat_logits = self(x_batch)    # Aladdin's vid #2 (5:06) says we can do --- self.forward(**batch)
        loss = criterion(yhat_logits, y_batch)
        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        acc = self.acc(yhat_logits.argmax(1), y_batch)
        self.log(f"train_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x_batch, y_batch = batch
        yhat_logits = self(x_batch)
        loss = criterion(yhat_logits, y_batch)
        self.log(f"val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        acc = self.acc(yhat_logits.argmax(1), y_batch)
        self.log(f"val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # def configure_optimizers(self):    # see Aladdin's vid #2 (7:55)
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)    # Q - can we include scheduler in here ?

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
        optimizer = torch.optim.Adam(params = self.parameters(), lr = self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                          optimizer, mode='min', factor=0.1, patience=3
                                                              )
        return {
              'optimizer': optimizer,
              'lr_scheduler': {
                  'scheduler': scheduler,
                  #'monitor': 'val_loss',   # NEED TO FIX THIS --- got error in .fit() on re-loading model & using scheduler, saying --- metric "val_loss" is not available! --- how come ????
                  'monitor': 'train_loss',
              }
          }


best_model = NN(cnn)   # so best_model is a Pytorch Lightning model !       # so best_model is a Pytorch Lightning model !

#****** LOAD PRE-TRAINED WEIGHTS ON TO ABOVE PYTORCH LIGHTNING MODEL ******
# (this model can be used only to run INFERENCE using the trained model ie. not for continued training from saved checkpoint)

checkpoint_path = "/content/gdrive/MyDrive/Colab Notebooks/VIT/Nov25/Resnet/BEST_MODEL_resnet34_lr=1e-05_epoch=8_val_loss=0.44_val_acc=0.88.ckpt"
checkpoint = torch.load(checkpoint_path)   
best_model.load_state_dict(checkpoint['state_dict'])
best_model = best_model.to(device)

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
            from torchvision.transforms import ToTensor
            final_image = ToTensor()(img_)
            final_image = final_image.unsqueeze(0)    # final_image.shape ---> torch.Size([1, 3, 224, 224])   
            final_image = final_image.to(device)

            # Run prediction on trained ViT model: 
            best_model.eval()
            with torch.no_grad():
              output = best_model(final_image)   # note that 'output' is a dictionary with 1 key ie 'logits'

# output ---> ImageClassifierOutput(loss=None, 
#                                   logits=tensor([[-0.6122, -0.7214,  3.5567, -0.4223, -0.0570, -0.4900, -0.1830, -0.6580]], device='cuda:0'), 
#                                   hidden_states=None, attentions=None)

            output_class = output.argmax(dim = 1).detach().cpu().item()
            id2label = {0:'Normal', 1:'Diabetes', 2:'Glaucoma', 3:'Cataract', 4:'Age related Macular Degeneration', 5:'Hypertension', 6:'Pathological Myopia', 7:'Other diseases/abnormalities'}
            prediction = id2label[output_class]     # returns for eg. --- 'Glaucoma'
            st.write(f'Prediction: {prediction}')

            # Ground Truth: --- need to do this
            # Need to add that bar chart from "3___Mlflow__basic_concise_code.ipynb" file in "Pytorch/Learning_Path" folder on Lptp

    else:    # this is for the 'About' tab on the Streamlit web page I guess
      st.subheader("About")
      st.info("Built with Streamlit")
      st.info("Jesus Saves")

if __name__ == '__main__':
    main()
