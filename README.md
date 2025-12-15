# CLIP Image Similarity Tool
A simple Python tool that compares two images using **[OpenCLIP](https://github.com/mlfoundations/open_clip)** and outputs a similarity score and percentage.

## Installation
```bash
python -m venv venv
Set-ExecutionPolicy Unrestricted -Scope Process
venv/Scripts/activate
pip install -r requirements.txt
```
## Building a standalone executable
If you want to use the tool without a Python installation, you can package it into a single executable using PyInstaller.

**With Console/Terminal:**
```bash
pip install pyinstaller
py -m PyInstaller compare.py --onefile --collect-all open_clip --collect-all open_clip_torch-3.2.0.dist-info --collect-all open_clip_train
```
**Without Console/Terminal:**
```bash
pip install pyinstaller
py -m PyInstaller compare.py --onefile --collect-all open_clip --collect-all open_clip_torch-3.2.0.dist-info --collect-all open_clip_train --noconsole
```
## Usage
```
python compare.py image1.png image2.png
```
OR 
```
python compare.py
```
(will ask for image paths to compare)

## Additional Information

Models by default are downloaded in C:/Users/username/.cache/huggingface/hub
