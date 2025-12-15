import sys
import torch
import open_clip as openClip
from PIL import Image

def loadModel(): # function for loading clip models
    device = "cuda" if torch.cuda.is_available() else "cpu" # sets the device that does the ai stuff - if user has a gpu with cuda cores then use those, otherwise do the calcs with cpu
    model, preprocess, _ = openClip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k") # basically js chooses the model and training data (copied this STRAIGHT from that one repository)
    return device, model, preprocess

def embedImage(imagePath, device, model, preprocess): # takes the image that is input input and turns into embed
    image = Image.open(imagePath).convert("RGB") # gets image from path and then turns into RGB format
    tensorImage = preprocess(image).unsqueeze(0).to(device) # resizes, crops and turns image into an image Tensor (an image as some numbers) then .unsqueeze turns the image tensor into batches or sm cus it needs to be like that for it to be put through the model, then .to(device) moves the image to where the model is (on the cpu or on gpu cuda cores)
    with torch.no_grad(): # tells torch that not doing data training
        embedding = model.encode_image(tensorImage) # finally runs the image through the CLIP model
        embedding = embedding / embedding.norm(dim=-1, keepdim=True) # i saw open clip repo did this in their usage demonstration so i copy
    return embedding
    # for your information, "In the context of AI, an embedding is a way of representing complex data (like words, images, or audio) as a list of numbers, known as a vector. This numerical format allows computers to "understand" and compare the meaning of different data points." Google AI Overview :D

def menuStatus():
    shouldMenu: bool
    if len(sys.argv) == 3: # if correct use of arguments
        shouldMenu = False
        main(shouldMenu)
    else:
        shouldMenu = True
        main(shouldMenu)
        

def main(shouldMenu):
    if shouldMenu == False:
        print("Loading CLIP model (first run ever may take a while because downloading the model from huggingface - please chill)...")
        imagePath1 = sys.argv[1] # set our image path to be the image path in arguments
        imagePath2 = sys.argv[2] # set our image path to be the image path in arguments
        process(imagePath1, imagePath2)
    else:
        imagePath1 = input("Path of first image? (no quotatiion marks)")
        imagePath2 = input("Path of second image? (no quotation marks)")
        process(imagePath1, imagePath2)

def process(imagePath1, imagePath2):
    device, model, preprocess = loadModel() # set our device (cpu or cuda cores) set our model and set our training data - all to whatever is in loadModel() - later can allow user to choose
    
    embeddedImage1 = embedImage(imagePath1, device, model, preprocess) # turn our images into embedded images
    embeddedImage2 = embedImage(imagePath2, device, model, preprocess) # turn our images into embedded images

    similarity = (embeddedImage1 @ embeddedImage2.T).item() # calculate image similarity score (no idea how)
    similarityPercentage = round(similarity*100)
    print("Device:", device)
    print("Image Similarity:", similarity)
    print("Image Similarity:", str(similarityPercentage) + "%")
    print(f"FINAL: {similarity}")
    



if __name__ == "__main__": #if run directly, not imported
    menuStatus()

if __name__ == "compare": # if imported as "compare.py" or sm like that
    pass


# __name__ is a built in variable for all python scripts, and if its __main__ then that means the python file was run directly but if this script is imported then __name__ will be the name of the script (compare)
# use this for when want to change model installation path
# import os
# os.environ["TORCH_HOME"] = r"path for where you want models to be downloaded"

# for the record, this is not vibecoded... (mostly)
# i use comments on each line so its easy to understand what happening