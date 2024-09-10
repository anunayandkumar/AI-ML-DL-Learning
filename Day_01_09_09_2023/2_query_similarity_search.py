from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import faiss
import numpy as np
import os

# Load the pre-trained model
model = resnet50(pretrained=True)
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the FAISS index
index = faiss.read_index('image_index.index')

# Function to find similar images
def find_similar_images(query_image_path, top_k=2):
    # Load and preprocess the query image
    query_image = Image.open(query_image_path)
    query_tensor = transform(query_image).unsqueeze(0)
    
    # Get the embedding for the query image
    with torch.no_grad():
        query_features = model(query_tensor)
        query_embedding = query_features.numpy().flatten()
    
    # Search for similar images
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    return distances, indices

# Example usage
query_image_path = 'query_image.jpeg'  
distances, indices = find_similar_images(query_image_path, top_k=2)

# Print out results
print(f"Top 2 similar images for {query_image_path}:")
# Extract the lists
distances_list = distances[0]
indices_list = indices[0]

# Iterate through the lists manually
for i in range(len(distances_list)):
    distance = distances_list[i]
    index = indices_list[i]
    print(f"Rank {i+1}: Image Index {index}, Distance: {distance}")
