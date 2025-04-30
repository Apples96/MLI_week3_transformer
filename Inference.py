import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import random
from Training_encoder import Load_MNIST_dataset
from datetime import datetime
from Training_encoder import VisionTransformerEncoder, Tokenize_and_CLS_img
from Training_encoder_decoder import Load_multi_digit_MNIST_dataset, VisionTransformerEncoderDecoder, Tokenize_multidigit_img, DigitTokenizer



# ENCODER MODEL to try out 10 random eamples

def inference_encoder_decoder_model():
    #Load Test dataset

    train_dataset, test_dataset = Load_multi_digit_MNIST_dataset()
    
    # Get 10 random examples of images with a digit each
    random_indices = random.sample(range(len(test_dataset)), 10)

    # Load model
    model = VisionTransformerEncoderDecoder()
    model_state_dict = torch.load('models/encoder_decoder_model_multi_digits_without_blanks_2025-04-30 19:17:15.488473.pt')
    model.load_state_dict(model_state_dict)
    correct = 0

    
    # Print out the digits and run the model to output predictions over the images
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(random_indices):
        img, labels = test_dataset[idx]
        
        # Tokenize and flatten image into patches
        img_patches, patches_pos, img_height, img_width = Tokenize_multidigit_img(img)
        digit_tokenizer = DigitTokenizer()
        decoder_input = digit_tokenizer.create_decoder_input(labels)
        decoder_target = digit_tokenizer.create_decoder_target(labels)
        # Run model to predict digits
        output_logits = model(img_patches, decoder_input)
        pred = F.softmax(output_logits, dim=1).argmax(dim=1)
        correct += (pred == decoder_target).sum().item()

        # Denormalize for visualization
        img = img * 0.3081 + 0.1307
        plt.subplot(3, 4, i+1)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Label: {labels}")
        plt.axis('off')

        print(f"Image {i+1}: True label = {labels}, Predicted = {pred}")

    plt.tight_layout()
    plt.savefig(f'mnist_samples_{datetime.now()}.png')
    print(f"\nSaved 10 random examples to 'mnist_samples_{datetime.now()}.png'")

    # Print accruancy of the predictions vs actual digit images
    print(f'Accuracy: {correct}/{len(random_indices)*17} ' # 17 tokens to be predicted per image
        f'({100. * correct / (len(random_indices)*17):.2f}%)') # 17 tokens to be predicted per image
    









# ENCODER MODEL to try out 10 random eamples

def inference_encoder_model():
    #Load Test dataset

    train_dataset, test_dataset = Load_MNIST_dataset()
    
    # Get 10 random examples of images with a digit each
    random_indices = random.sample(range(len(test_dataset)), 10)

    # Load model
    model = VisionTransformerEncoder()
    model_state_dict = torch.load('models/encoder_model_single_digits_2025-04-30 19:57:46.331278.pt')
    model.load_state_dict(model_state_dict)
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






if __name__ == "__main__":
    print("ENCODER MODEL ON SINGLE DIGITS")
    inference_encoder_model()
    
    print("ENCODER DECODER MODEL ON MULTI DIGITS")
    inference_encoder_decoder_model()



