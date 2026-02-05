pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
# Import necessary libraries
import torch
import clip
import os
import random
import pandas as pd
from PIL import Image
from IPython.display import display
from google.colab import files

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load labels from CSV
csv_path = "/content/labels.csv"
df = pd.read_csv(csv_path)
labels = df['Label'].tolist()

# Upload images to Colab
print("Please upload images (JPG, JPEG, PNG, WEBP, GIF, BMP) for classification...")
uploaded_files = files.upload()

# Get list of supported image files
supported_formats = (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp")
image_files = [file for file in uploaded_files.keys() if file.lower().endswith(supported_formats)]

# Ensure images are available
if not image_files:
    raise ValueError("No valid image files were uploaded. Please upload at least one.")

# Select a random image
random_image_path = random.choice(image_files)
print(f"\nSelected Random Image: {random_image_path}")
image = Image.open(random_image_path).convert("RGB")
image_processed = preprocess(image).unsqueeze(0).to(device)

# Display the selected image
# image.show()
text_inputs = clip.tokenize(labels).to(device)
print(f"\nDisplaying Selected Image: {random_image_path}")
display(image)
# Perform zero-shot classification
with torch.no_grad():
    image_features = model.encode_image(image_processed)
    text_features = model.encode_text(text_inputs)
    similarity = torch.cosine_similarity(image_features, text_features, dim=-1)

# Get the best match
best_match = labels[similarity.argmax().item()]
print(f"\nPredicted Label: {best_match}")

# Display similarity scores for all labels
print("\nSimilarity Scores:")
for label, score in zip(labels, similarity.tolist()):
    print(f"{label}: {score:.4f}")


