# Image Classification using CLIP

This project implements a zero-shot image classification system using OpenAI's CLIP (Contrastive Language-Image Pretraining) model in Google Colab. The system predicts the most suitable label for a given image by comparing it with a set of predefined labels from a CSV file.

## Features
- Uses OpenAI's CLIP model for image classification.
- Supports zero-shot classification.
- Accepts image uploads in formats like JPG, JPEG, PNG, WEBP, GIF, and BMP.
- Provides similarity scores for all labels.

## Prerequisites
- Python 3.x
- Google Colab
- `torch`, `torchvision`, `clip`, `pandas`, `PIL`, and `IPython`

## Installation

Run the following commands in your Colab notebook:

```bash
!pip install torch torchvision
!pip install git+https://github.com/openai/CLIP.git
!pip install pandas
```

## Project Structure
```
.
 labels.csv           # CSV containing labels for classification
 Image Classification.ipynb  # Colab notebook with implementation
 README.md            # Project documentation
```

## How to Use

1. **Upload Your CSV**: Ensure your `labels.csv` file contains a column named `Label` with classification labels.

2. **Upload Images**: The script supports image uploads using Colab's `files.upload()`.

3. **Run the Code**: Execute the notebook cells step by step.

4. **View Results**: The model will display the uploaded image and print the predicted label along with similarity scores for all labels.

## Example CSV Format
```
Label
Dog
Cat
Car
Tree
```

## Troubleshooting
- **CUDA Error**: Ensure GPU is enabled by navigating to `Runtime` -> `Change runtime type` -> `GPU`.
- **File Upload Error**: Ensure images are in the supported formats (JPG, JPEG, PNG, WEBP, GIF, BMP).
- **Label Loading Error**: Confirm the `labels.csv` has a `Label` column.
