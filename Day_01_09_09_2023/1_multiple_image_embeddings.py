from PIL import Image                              # For image handling and all like loading
import torch                                       # Deep learning library 
import torchvision.transforms as transforms        # For image preprocessing
from torchvision.models import resnet50            # ResNet50 is a pre-trained model based on CNN architecture
import faiss                                       # Facebook AI Similarity Search library, used for vector search.
import numpy as np                                 # For array manipulation tasks
import os                                          # For directory operations


# Load the pre-trained model
model = resnet50(pretrained=True)                  
model.eval()                                       # Sets the evaluation mode because some layers behave differently

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),                       
    transforms.CenterCrop(224),
    transforms.ToTensor(),                         # Converts the image to a 2D multi-dimensional array
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Directory containing images
image_directory = 'images/'  

# Initialize a list to store image embeddings
embeddings = []

# Iterate over all images in the directory
for filename in os.listdir(image_directory):
    if filename.endswith('.jpeg') or filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Get the image embedding
        with torch.no_grad():  # Disable gradient calculation
            features = model(image_tensor)
            embedding = features.numpy().flatten()  # Convert tensor to numpy array and flatten it
            embeddings.append(embedding)

# Convert list of embeddings to a numpy array
embeddings = np.array(embeddings)

# Create and populate a FAISS index
dimension = embeddings.shape[1]  # Length of the embedding vector
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add all embeddings to the index

# Save the index to a file
faiss.write_index(index, 'image_index.index')

# To check the number of stored embeddings
print("Number of stored embeddings:", embeddings.shape[0])
