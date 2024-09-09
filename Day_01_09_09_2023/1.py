from PIL import Image                              #for image handling and all like loading
import torch                                       #deep learning library 
import torchvision.transforms as transforms        #for image preprocessing
from torchvision.models import resnet50            #resnet50 is pre trained model based on cnn architecture ?
import faiss                                       #facebook AI Similarity Search library, used for vector search.
import numpy as np                                 #for array manipulation tasks



# Load the pre-trained model
model = resnet50(pretrained=True)                  
model.eval()                                       #sets the evaluation mode because some layers behave differently



# Defining the image transformations
transform = transforms.Compose([
    transforms.Resize(256),                       
    transforms.CenterCrop(224),
    transforms.ToTensor(),                         #will convert the image in 2 D multi dimensional array (note this numerical representation is diffrenet as vector embedding's numerical representation.The earlier one is just to pre process having dimensions , order of height , width in some order in accordance with requirements so that here dl model can process but later one is based on feature extraction .)
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




image_path = 'images/1.jpeg'  
image = Image.open(image_path)
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension as tranform accepts batch of images though I have 0 image




# Getting the image embedding
with torch.no_grad():                           #disabling gradient calculation
    features = model(image_tensor)
    # Use the output of the last layer as the image embedding
    embedding = features.numpy().flatten()     #tenstor to 2d numpy array and then flatten will further convert that to 1d array 




# Create and populate a FAISS index
dimension = embedding.shape[0]             #length of the embedding vector
index = faiss.IndexFlatL2(dimension)
index.add(np.array([embedding]))           #index.add(np.array([embedding])): Adds the embedding vector to the index. The np.array([embedding]) ensures the embedding is in the right format (2D array with one row).

# Save the index to a file (optional)
faiss.write_index(index, 'image_index.index')

# To check the stored embedding
print("Stored embedding:", embedding)

# Load the index (if needed)
# index = faiss.read_index('image_index.index')
