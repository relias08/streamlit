# All Pytorch imports for Computer Vision

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

id2label = {0:'Normal', 1:'Diabetes', 2:'Glaucoma', 3:'Cataract', 4:'Age related Macular Degeneration', 5:'Hypertension', 6:'Pathological Myopia', 7:'Other diseases/abnormalities'}
label2id = {v:k for k,v in id2label.items()}

from transformers import ViTFeatureExtractor, ViTForImageClassification

model_name = 'google/vit-base-patch16-224-in21k'
num_classes = 8

feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name,
                                num_labels = num_classes,         # num_classes = 8
                                label2id = label2id,
                                id2label = id2label)  #.to(device)   

# Ritheesh did not use .to(device) --- this is applied inside Trainer automatically!

# ---------------------------------------------------------------------------------
#****** Build the Pytorch Lightning model ******
# Nice if I could put this block into a separate file called model.py and then use --- 'from model import Classifier'

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


best_model = Classifier(model, lr=2e-5)     # so best_model is a Pytorch Lightning model !

#****** LOAD PRE-TRAINED WEIGHTS ON TO ABOVE PYTORCH LIGHTNING MODEL ******
# (this model can be used only to run INFERENCE using the trained model ie. not for continued training from saved checkpoint)

checkpoint_path = "/content/gdrive/MyDrive/Colab Notebooks/_CNN___Main/_____ViT/tb_logs123/Test___June_19/version_3/checkpoints/epoch=15-step=880.ckpt"

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
            img = Image.open(image_file)         # jpg img => PIL Object --- img.size -> (1024, 768)
            img = img.resize((256, 256))      

            # -----------------------------
            # For displaying input image by itself in Streamlit
            img_ = img.copy()     # Note - img_ is a PIL Obj & st.image() below requires a PIL obj to display it on the Streamlit web page !!!
            st.write('Input image')
            st.image(img_, width=250)
          

            # ****** MAKE PREDICTION ******
            # First pre-process the input image using feature_extractor (ie. normalize + re-size):
            processed_image = feature_extractor(img_, return_tensors = 'pt')  # processed_image is a dict with 1 key ie. 'pixel_values'
            final_image = processed_image['pixel_values'][0].unsqueeze(0)    # final_image.shape ---> torch.Size([1, 3, 224, 224])
            final_image = final_image.to(device)

            # Run prediction on trained ViT model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            best_model = best_model.to(device)

            best_model.eval()
            with torch.no_grad():
              output = best_model(final_image)   # note that 'output' is a dictionary with 1 key ie 'logits'

# output ---> ImageClassifierOutput(loss=None, 
#                                   logits=tensor([[-0.6122, -0.7214,  3.5567, -0.4223, -0.0570, -0.4900, -0.1830, -0.6580]], device='cuda:0'), 
#                                   hidden_states=None, attentions=None)

            class = output.logits.argmax(dim = 1)
            prediction = id2label[class.detach().cpu().item()]     # returns for eg. --- 'Glaucoma'
            st.write(f'Prediction: {prediction}')

            # Ground Truth:


####################################################################
# # ****** Pre-process the input image using feature_extractor ******

# # Label=2 (because it is from folder 2) ie. Glaucoma
# test_image_path = "/content/gdrive/MyDrive/Colab Notebooks/_CNN___Main/_____ViT/test_/2/1020_left.jpg"

# # Label=5  ie. Hypertension
# # test_image_path = "/content/gdrive/MyDrive/Colab Notebooks/_CNN___Main/_____ViT/test_/5/1031_right.jpg"

# test_image = PIL.Image.open(test_image_path)    # test_image.size ---> (512, 512)

# processed_image = feature_extractor(test_image, return_tensors = 'pt')  # processed_image is a dict with 1 key ie. 'pixel_values'
# final_image = processed_image['pixel_values'][0].unsqueeze(0)    # final_image.shape ---> torch.Size([1, 3, 224, 224])
# final_image = final_image.to(device)

# # ---------------------------------------------------------------------------------
# #****** Run prediction on trained ViT model ******

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# best_model = best_model.to(device)

# best_model.eval()
# with torch.no_grad():
#   output = best_model(final_image)   # note that 'output' is a dictionary with 1 key ie 'logits'

# # output ---> ImageClassifierOutput(loss=None, logits=tensor([[-0.6122, -0.7214,  3.5567, -0.4223, -0.0570, -0.4900, -0.1830, -0.6580]],
# #                                   device='cuda:0'), hidden_states=None, attentions=None)

# prediction = output.logits.argmax(dim = 1)

# id2label[prediction.detach().cpu().item()]     # returns for eg. --- 'Glaucoma'
