import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from Training_encoder import Load_MNIST_dataset
from Training_encoder import VisionTransformerEncoder, Tokenize_and_CLS_img


# ENCODER MODEL to try out 10 random eamples

# 4. Visualize 10 random examples
def inference_encoder_model():
    #Load Test dataset

    train_dataset, test_dataset = Load_MNIST_dataset()
    
    # Get 10 random examples of images with a digit each
    random_indices = random.sample(range(len(test_dataset)), 10)

    # Load model
    model = VisionTransformerEncoder()
    model_state_dict = torch.load('models/encoder_model_single_digits_2025-04-29 12:23:01.213989.pt')
    model.load_state_dict(model_state_dict)
    predicted_digits = []
    correct = 0

    
    # Print out the digits and run the model to output predictions over the images
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(random_indices):
        img, label = test_dataset[idx]
        
        # Tokenize and flatten image into patches
        img_patches, patches_pos, img_height, img_width = Tokenize_and_CLS_img(img)
        # Run model to predict digits
        CLStoken_logits, encoder_hidden_state = model(img_patches)
        pred = F.softmax(CLStoken_logits, dim=0).argmax(dim=0)
        correct += (pred == label).sum().item()

        # Denormalize for visualization
        img = img * 0.3081 + 0.1307
        plt.subplot(3, 4, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')

        print(f"Image {i+1}: True label = {label}, Predicted = {pred}")

    plt.tight_layout()
    plt.savefig('mnist_samples.png')
    print("\nSaved 10 random examples to 'mnist_samples.png'")

    # Print accruancy of the predictions vs actual digit images
    print(f'Accuracy: {correct}/{len(random_indices)} '
        f'({100. * correct / len(random_indices):.2f}%)')


# ENCODER MODEL to try out 10 random eamples


if __name__ == "__main__":
    inference_encoder_model()



