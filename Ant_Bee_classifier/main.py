from model import CNN
from torchvision import transforms
import torch
from PIL import Image
model=CNN()
model.load_state_dict(torch.load('My_trained_CNN_(ant_vs_bee)_model.pth'))
model.eval()
labels=['Bee','Ant']
def classify(image):
    trans=transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image=trans(image)
    prediction=labels[torch.argmax(model(image)).item()]
    return prediction
    
print("<<<<<<<<<<<<Let's Predict>>>>>>>>>>>>")
while True:
    image=input('Give the image path : ')
    if image.lower()=="quit":
        break
    if image.lower()=='exit':
        break
    try:
        image=Image.open(image)
        print(f"The prediction is {classify(image)}")
    except:
        print(f'Could not open {image}')