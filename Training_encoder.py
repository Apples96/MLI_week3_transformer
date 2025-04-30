from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch import nn, optim
from Model import VisionTransformerEncoder
from datetime import datetime




#ENV VARIABLES

# BATCH_SIZE = 


###### TRAINING ON IMAGES WITH SINGLE DIGITS

#1 Load MNIST_dataset as tensors and normalise with known MNIST dataset mean (0.1307) and standard deviation (0.3081)

def Load_MNIST_dataset():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) # these are the MNIST mean and standard deviation. 
        ])
        
    # Download and load training data
    train_dataset = datasets.MNIST(root='./data', 
                                    train=True, 
                                    download=True, 
                                    transform=transform)

    # Download and load test data
    test_dataset = datasets.MNIST(root='./data', 
                                    train=False, 
                                    download=True, 
                                    transform=transform)

    return train_dataset, test_dataset


#2 Create an image tokenizer, that will be used when processing the data

def Tokenize_and_CLS_img(img, patch_size = 14): # assumes that
        _, img_height, img_width = img.shape # MNIST images are 1×28×28 (channels, height, width) - this takes shape of the first image in the batch.

        # Calculate number of patches in each dimension
        if img_height % patch_size == 0:
            num_patches_h = img_height // patch_size
        else:
            num_patches_h = img_height // patch_size +1
        if img_width % patch_size == 0:
            num_patches_w = img_width // patch_size
        else:
            num_patches_w = img_width // patch_size +1

        # Create empty tensor to store patches and positions
        CLStoken = torch.zeros(196)
        img_patches = [CLStoken]
        patches_pos = [0]
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_i_j_vector = img[:, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)].squeeze(0).reshape(196)
                img_patches.append(patch_i_j_vector)
                patch_pos = (i, j) # positions start at 1 since CLStoken is pos 0
                patches_pos.append(patch_pos)
        img_patches = torch.stack(img_patches)

        return img_patches, patches_pos, img_height, img_width
    
#3 Train the data

def train_encoder_model():
    # Load MNIST dataset and data loaders
    train_dataset, test_dataset = Load_MNIST_dataset()
    
    # Create dataloaders that load batches of tokenized MNIST image patches (batch_size, 4, 196) NB 196 = 14*14
    # tokenized_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = cpu_count())
    # tokenized_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = cpu_count())

    # Initialize Encoder model, loss funtcion and optimizer
    model = VisionTransformerEncoder()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Training
    epochs = 3
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for idx in range(len(train_dataset)):
            img, label = train_dataset[idx]
            label_tensor = torch.tensor(label)
            optimizer.zero_grad()
            img_patches, patches_pos, img_height, img_width = Tokenize_and_CLS_img(img)
            CLStoken_logits, encoder_hidden_state = model(img_patches)
            loss = criterion(CLStoken_logits, label_tensor)
            epoch_train_loss += loss.item()
            loss.backward()
            optimizer.step()
              
            # Print stats every 100 batches
            if idx % 10000 == 0: 
                print(f'Train Epoch: {epoch} [{idx}/{len(train_dataset)} '
                      f'({100. * idx / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}')
        avg_train_loss = epoch_train_loss / len(train_dataset)
        
        
        # Test model after each epoch and print stats
        model.eval()
        epoch_test_loss = 0
        correct = 0
        for idx in range(len(test_dataset)):
            img, label = test_dataset[idx]
            label_tensor = torch.tensor(label)
            img_patches, patches_pos, img_height, img_width = Tokenize_and_CLS_img(img)
            CLStoken_logits, encoder_hidden_state = model(img_patches)
            loss = criterion(CLStoken_logits, label_tensor)
            epoch_test_loss += loss.item()
            pred = F.softmax(CLStoken_logits, dim=0).argmax(dim=0, keepdim=True)
            correct += (pred == label_tensor.view_as(pred)).sum().item()
        avg_test_loss = epoch_test_loss/len(test_dataset)
    
        print(f'Train set: Average loss: {avg_train_loss:.4f}, '
            f'Test set: Average loss: {avg_test_loss:.4f}, '
            f'Accuracy: {correct}/{len(test_dataset)} '
            f'({100. * correct / len(test_dataset):.2f}%)')
        
        # Save the trained model
        torch.save(model.state_dict(), f'models/encoder_model_{datetime.now()}.pt') #Saves the trained model's parameters (weights and biases) 



if __name__ == "__main__":
    train_encoder_model()