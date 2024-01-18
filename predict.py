import argparse
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
from model import custom_model

def make_args():
    parser = argparse.ArgumentParser(description='Inference script')
    parser.add_argument('--checkpoint', type=str, help='Path to the model checkpoint', default='checkpoint_model.pth')
    parser.add_argument('--image_path', type=str, help='Path to the image you want to predict')
    parser.add_argument('--json_path', type=str, help='Path to the JSON file containing class names')
    parser.add_argument('--top_k', type=int, help='Top k classes to display', default=5)
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    return parser.parse_args()

def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image)
    img = transform(img)
    return img

def predict(image_path, model, topk=5):
    model.eval()

    img = process_image(image_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(img)

    probs = torch.exp(log_probs)
    top_probs, top_indices = probs.topk(topk, dim=1)

    return top_probs, top_indices

def display_prediction(image_path, model, json_path, topk=5):
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    top_probs, top_indices = predict(image_path, model, topk)

    top_probs = top_probs[0].cpu().numpy()
    top_indices = top_indices[0].cpu().numpy()

    class_names = [cat_to_name[str(idx + 1)] for idx in top_indices]

    print("Top Predictions:")
    for i in range(topk):
        print(f"{class_names[i]}: {top_probs[i] * 100:.2f}%")

def main():
    args = make_args()
    if args.gpu and not torch.cuda.is_available():
        print("GPU is not available. Using CPU.")
        args.gpu = False

    checkpoint = torch.load(args.checkpoint, map_location='cpu')  # Adjust for compatibility
    model_load = custom_model()
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model_load.to(device)
    model_load.class_to_idx = checkpoint['class_to_idx']
    model_load.load_state_dict(checkpoint['state_dict'])

    display_prediction(args.image_path, model_load, args.json_path, args.top_k)

if __name__ == '__main__':
    main()
