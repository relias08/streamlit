# Entire code worked perfectly fine !

# To launch the Streamlit web page, the current app.py file is called inside the 
# [Deploy Fundus model using Streamlit --- Colab + ngrok.ipynb](https://colab.research.google.com/drive/1W2VmuhsIKEwiAJQoSc0kaZKmOk96DSV8#scrollTo=8UXnKWbBOtGQ) 
# file stored in Google Drive using the following line of code:
# !streamlit run https://github.com/relias08/streamlit/blob/main/Streamlit_ResNet_2_app.py&>/dev/null&

# IMPORTANT NOTE --- in this file we are running predictions using model(input) and not requests.post() 
# and that RestAPI end-point from Mlflow !

# This file is based on --- "jcharis___mitochondria___app.py"

# Important Points:
# - st.image() which displays an image on the Streamlit web page requires PIL objects !!!
# - never use .astype('int') --- always use .astype(np.uint8)   # found out the hard way!   Big Q - what is the difference ???


# ---------------------------------------------------------------------------------
# Imports 

import torch
import pytorch_lightning as pl
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------------------------------------------------------
#****** Get the best model using torch.load() ******

class NN(pl.LightningModule):
    def __init__(self, model, lr: float = 1e-5, **kwargs):
        super().__init__()
        self.save_hyperparameters('lr', *list(kwargs))
        self.model = model
        # self.model_used = mod    # Q - will this save the mod in mlflow ui? (just like I once saved data used - small_data)
        self.forward = self.model.forward
        self.acc = Accuracy(task='multiclass',
                            num_classes=8)

stored_model = "/content/gdrive/MyDrive/Colab Notebooks/VIT/model_88.pth"
loaded_best_model = torch.load(stored_model, map_location = device)

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
            loaded_best_model.eval()
            with torch.no_grad():
              output = loaded_best_model(final_image)   # note that 'output' is a dictionary with 1 key ie 'logits'

# output ---> ImageClassifierOutput(loss=None, 
#                                   logits=tensor([[-0.6122, -0.7214,  3.5567, -0.4223, -0.0570, -0.4900, -0.1830, -0.6580]], device='cuda:0'), 
#                                   hidden_states=None, attentions=None)

            output_class = output.argmax(dim = 1).detach().cpu().item()
            id2label = {0:'Normal', 1:'Diabetes', 2:'Glaucoma', 3:'Cataract', 4:'Age related Macular Degeneration', 5:'Hypertension', 6:'Pathological Myopia', 7:'Other diseases/abnormalities'}
            prediction = id2label[output_class]     # returns for eg. --- 'Glaucoma'
            st.write(f'Prediction: {prediction}')

            # Ground Truth: --- need to do this

    else:    # this is for the 'About' tab on the Streamlit web page I guess
      st.subheader("About")
      st.info("Built with Streamlit")
      st.info("Jesus Saves")

if __name__ == '__main__':
    main()
