import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms, models
from collections import OrderedDict
from model import custom_model


def make_args():
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    parser.add_argument('--arch', type=str, help='Model trainning', default='vgg16')
    parser.add_argument('--learning_rate', type=float, default=0.002, help='Learning rate for trainning')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number hidden units')
    parser.add_argument('--dropout', type=float, help='Model trainning', default=0.05)
    parser.add_argument('--epochs', type=int, default=5, help='Number of train epochs')
    parser.add_argument('--gpu',action='store_true', default='gpu', help='Use GPU for training')
    
    return parser.parse_args()

def load_data(data_path='flowers'):
    
    data_dir = data_path
    train_dir= data_dir + '/train'
    valid_dir= data_dir + '/valid'
    test_dir= data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([
        transforms.RandomRotation(25),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, shuffle = False)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, shuffle = False)

    return train_data, trainloader, testloader, validloader

def main():
    args = make_args()
    if not os.path.exists(args.data_dir):
        print(f"Directory {args.data_dir} does not exist!")
        return
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    train_data, trainloader, validloader, testloader = load_data(args.data_dir)
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    model = custom_model(backbone=args.arch, hidden_units=args.hidden_units, dropout=args.dropout).to(device)
    # Define the loss function
    criterion = nn.NLLLoss()
    # criterion = nn.CrossEntropyLoss()
    # Define the optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)


    print("--------------------------------------< Model >--------------------------------------------")
    print(model)
    print("--------------------------------------< Start Training >--------------------------------------------")
    epochs = args.epochs
    print_every = 20

    # Initialize variables for tracking loss and accuracy
    running_loss = 0
    running_accuracy = 0

    # Lists to store training and validation losses
    training_loss_data = []
    validating_loss_data = []

    # Loop through epochs
    for e in range(epochs):
        steps = 0
        model.train()  # Set the model to training mode
        # Iterate through training data
        for images, labels in trainloader:
            steps += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients

            loggs = model(images)
            train__loss = criterion(loggs, labels)
            train__loss.backward()
            optimizer.step()

            # Calculate metrics
            met = torch.exp(loggs)
            top_ps, top_class =met.topk(1, dim=1)
            matches = (top_class == labels.view(*top_class.shape)).type(torch.FloatTensor)
            accuracy = matches.mean()

            # Track metrics
            running_loss += train__loss.item()
            running_accuracy += accuracy.item()

            if steps % print_every == 0:
                # Validation phase
                model.eval()  # Set the model to evaluation mode
                with torch.no_grad():
                    validation_loss = 0
                    validation_accuracy = 0
                    for val_images, val_labels in validloader:
                        val_images, val_labels = val_images.to(device), val_labels.to(device)

                        val_log_ps = model(val_images)
                        val_loss = criterion(val_log_ps, val_labels)
                        val_ps = torch.exp(val_log_ps)
                        val_top_ps, val_top_class = val_ps.topk(1, dim=1)
                        val_matches = (val_top_class == val_labels.view(*val_top_class.shape)).type(torch.FloatTensor)
                        val_accuracy = val_matches.mean()

                        validation_loss += val_loss.item()
                        validation_accuracy += val_accuracy.item()

                # Calculate average loss and accuracy
                avg_train__loss = running_loss / print_every
                avg_train_accuracy = running_accuracy / print_every * 100
                avg_validation_loss = validation_loss / len(validloader)
                avg_validation_accuracy = validation_accuracy / len(validloader) * 100

                # Append values to lists
                training_loss_data.append(avg_train__loss)
                validating_loss_data.append(avg_validation_loss)

                # Print metrics
                print(f'Epoch {e+1}/{epochs} | Step {steps}')
                print(f'Running Training Loss: {avg_train__loss:.3f}')
                print(f'Running Training Accuracy: {avg_train_accuracy:.2f}%')
                print(f'Validation Loss: {avg_validation_loss:.3f}')
                print(f'Validation Accuracy: {avg_validation_accuracy:.2f}')

                # Reset metrics and set the model back to training mode
                running_loss = 0
                running_accuracy = 0
                model.train()

    print("--------------------------------------< Finished Training >--------------------------------------------")
    
    model = model.eval()  

    running_test_loss = 0
    running_test_accuracy = 0  

    with torch.no_grad():
        for test_images, test_labels in testloader:  
            test_images, test_labels = test_images.to(device), test_labels.to(device)

            # Ensure the model and input are on the same device
            model = model.to(device)

            test_outputs = model(test_images) 

            # Calculate test loss
            test_loss = criterion(test_outputs, test_labels)
            running_test_loss += test_loss.item()

            # Calculate test accuracy
            test_ps = torch.exp(test_outputs)
            top_p, top_class = test_ps.topk(1, dim=1)
            test_equals = top_class == test_labels.view(*top_class.shape)
            running_test_accuracy += torch.mean(test_equals.type(torch.FloatTensor)).item()

    # Calculate and print average test loss and accuracy
    average_test_loss = running_test_loss / len(testloader)
    test_accuracy = running_test_accuracy / len(testloader)

    print('Average Test Loss: {:.3f}'.format(average_test_loss))
    print('Test Accuracy: {:.3f}%'.format(test_accuracy * 100))

    
    # Save the checkpoint
    # Define the file path where you want to save the model
    filepath = 'checkpoint_model.pth'

    # Save the model's state dictionary
    model.class_to_idx = train_data.class_to_idx
    torch.save({'input_size': 25088,
                'output_size': 102,
                'structure': 'vgg16',
                'learning_rate': 0.002,
                'classifier': model.classifier,
                'epochs': epochs,
                'optimizer': optimizer.state_dict(),
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx},filepath)

if __name__ == '__main__':
    main()