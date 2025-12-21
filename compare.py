import sys
import torch
import time
import open_clip as openClip
from PIL import Image

def loadModel(): 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, _ = openClip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k") # basically js chooses the model and training data (copied this STRAIGHT from that one repository)
    return device, model, preprocess

def embedImage(imagePath, device, model, preprocess):
    image = Image.open(imagePath).convert("RGB")
    tensorImage = preprocess(image).unsqueeze(0).to(device) # resizes, crops and turns image into an image Tensor (an image as some numbers) then .unsqueeze turns the image tensor into batches or sm cus it needs to be like that for it to be put through the model
    with torch.no_grad():
        embedding = model.encode_image(tensorImage) # finally runs the image through the CLIP model
        embedding = embedding / embedding.norm(dim=-1, keepdim=True) # open clip repo did this in their usage demonstration so i copy
    return embedding

def menuStatus():
    shouldMenu: bool
    if len(sys.argv) == 3:
        shouldMenu = False
        main(shouldMenu)
    else:
        shouldMenu = True
        main(shouldMenu)
        

def main(shouldMenu):
    if shouldMenu == False:
        print("Loading CLIP model (first run ever may take a while because downloading the model from huggingface - please chill)...")
        imagePath1 = sys.argv[1] 
        imagePath2 = sys.argv[2] 
        process(imagePath1, imagePath2)
    else:
        imagePath1 = input("Path of first image? (no quotatiion marks)")
        imagePath2 = input("Path of second image? (no quotation marks)")
        process(imagePath1, imagePath2)

def process(imagePath1, imagePath2):
    device, model, preprocess = loadModel() 
    
    embeddedImage1 = embedImage(imagePath1, device, model, preprocess) 
    embeddedImage2 = embedImage(imagePath2, device, model, preprocess)

    similarity = (embeddedImage1 @ embeddedImage2.T).item() # (no idea what that @ does but it works)
    similarityPercentage = round(similarity*100)
    print("Device:", device)
    print("Image Similarity:", similarity)
    print("Image Similarity:", str(similarityPercentage) + "%")
    print(f"FINAL: {similarity}")
    time.sleep(5)



if __name__ == "__main__": #if run directly, not imported
    menuStatus()

if __name__ == "compare": # if imported as "compare.py" or sm like that
    pass
