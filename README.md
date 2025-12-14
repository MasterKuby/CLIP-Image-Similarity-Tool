# CLIP Image Similarity Tool
A simple Python tool that compares two images using **[OpenCLIP](https://github.com/mlfoundations/open_clip)** and outputs a similarity score and percentage.

## Installation
```bash
python -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
```
## Usage
```
python compare.py image.png referenceImage.png
```
OR 
```
python compare.py
```
(will ask for image paths to compare)

## Additional Information

Models by default are downloaded in C:/Users/username/.cache/huggingface/hub
