from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F
from torch import nn, optim
from Model import VisionTransformerEncoder, VisionTransformerEncoderDecoder
from datetime import datetime




#ENV VARIABLES

# BATCH_SIZE = 

###### TRAINING ON IMAGES WITH MULTIPLE DIGITS

#1 Load the multi digit MNIST_dataset (already normalized and as tensors)

class LoadedMultiDigitMNIST(Dataset):
    def __init__(self, data, labels):
        # Skip the parent class __init__ to avoid generating new data
        self.data = data
        self.labels = labels
        
    def __len__(self):
            return len(self.data)
            
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def Load_multi_digit_MNIST_dataset():
    train_file_path='./data/multi_digit_MNIST/train_dataset_multi_digit_MNIST_withoutblanks.pt'
    test_file_path='./data/multi_digit_MNIST/test_dataset_multi_digit_MNIST_withoutblanks.pt'
    train_data = torch.load(train_file_path)
    test_data = torch.load(test_file_path)

    train_multi_digit_MNIST_dataset = LoadedMultiDigitMNIST(train_data['data'], train_data['labels']) 
    test_multi_digit_MNIST_dataset = LoadedMultiDigitMNIST(test_data['data'], test_data['labels'])
    
    # Return an instance with the loaded data
    return train_multi_digit_MNIST_dataset, test_multi_digit_MNIST_dataset


#2 Create an image tokenizer and a labels tokenizer, that will be used when processing the data

def Tokenize_multidigit_img(img, patch_size = 14): # assumes that
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
        img_patches = []
        patches_pos = []
        
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch_i_j_vector = img[:, patch_size*i:patch_size*(i+1), patch_size*j:patch_size*(j+1)].squeeze(0).reshape(196)
                img_patches.append(patch_i_j_vector)
                patch_pos = (i, j) # positions start at 0
                patches_pos.append(patch_pos)
        img_patches = torch.stack(img_patches)

        return img_patches, patches_pos, img_height, img_width

class DigitTokenizer:
    def __init__(self):
        # Define special tokens
        self.SOS_token = 10  # Start of sequence token
        self.EOS_token = 11  # End of sequence token
        self.PAD_token = 12  # Padding token
        
        # Create token-to-digit and digit-to-token mappings
        self.token_to_digit = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            self.SOS_token: '<SOS>',
            self.EOS_token: '<EOS>',
            self.PAD_token: '<PAD>'
        }
        
        self.digit_to_token = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '<SOS>': self.SOS_token,
            '<EOS>': self.EOS_token,
            '<PAD>': self.PAD_token
        }
        
        # Vocabulary size (0-9 digits + special tokens)
        self.vocab_size = len(self.token_to_digit)
    
    def encode_digit_sequence(self, digit_sequence):
        """
        Convert a list of digits to token indices with SOS and EOS tokens
        
        Args:
            digit_sequence: List of digits (integers or strings)
            
        Returns:
            list: List of token indices [SOS, digit1, digit2, ..., EOS]
        """
        # Convert all digits to strings to ensure consistent lookup
        digit_sequence = [str(d) for d in digit_sequence]
        
        # Create token sequence with SOS at start and EOS at end
        tokens = [self.SOS_token]
        tokens.extend([self.digit_to_token[d] for d in digit_sequence])
        tokens.append(self.EOS_token)
        
        return tokens
    
    def decode_token_sequence(self, token_sequence):
        """
        Convert token indices back to digit sequence, removing special tokens
        
        Args:
            token_sequence: List of token indices
            
        Returns:
            list: List of digits (as strings)
        """
        # Filter out special tokens and convert to digits
        digits = []
        for token in token_sequence:
            if token not in [self.SOS_token, self.EOS_token, self.PAD_token]:
                digits.append(self.token_to_digit[token])
        
        return digits
    
    def create_decoder_input(self, digit_sequence, max_length=None):
        """
        Create input sequence for decoder training (right-shifted target sequence)
        
        Args:
            digit_sequence: List of digits
            max_length: Maximum sequence length (will pad if needed)
            
        Returns:
            tensor: Tensor containing [SOS, digit1, digit2, ...] (without EOS)
        """
        # Encode the sequence
        tokens = self.encode_digit_sequence(digit_sequence)
        
        # For decoder input, use tokens except the last (EOS)
        decoder_input = tokens[:-1]
        
        # Pad if needed
        if max_length and len(decoder_input) < max_length:
            decoder_input.extend([self.PAD_token] * (max_length - len(decoder_input)))
        
        return torch.tensor(decoder_input)
    
    def create_decoder_target(self, digit_sequence, max_length=None):
        """
        Create target sequence for decoder training
        
        Args:
            digit_sequence: List of digits
            max_length: Maximum sequence length (will pad if needed)
            
        Returns:
            tensor: Tensor containing [digit1, digit2, ..., EOS] (without SOS)
        """
        # Encode the sequence
        tokens = self.encode_digit_sequence(digit_sequence)
        
        # For target, use tokens except the first (SOS)
        decoder_target = tokens[1:]
        
        # Pad if needed
        if max_length and len(decoder_target) < max_length:
            decoder_target.extend([self.PAD_token] * (max_length - len(decoder_target)))
        
        return torch.tensor(decoder_target)



#3 Train the data

def train_multidigit_encoder_decoder_model(test = False):
    # Load MNIST dataset and data loaders
    train_dataset, test_dataset = Load_multi_digit_MNIST_dataset()
    if test == True:
        train_dataset = train_dataset[:10]
        test_dataset = test_dataset[:10]
    
    
    # Create dataloaders that load batches of tokenized MNIST image patches (batch_size, 4, 196) NB 196 = 14*14
    # tokenized_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers = cpu_count())
    # tokenized_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = cpu_count())

    # Initialize Encoder model, loss funtcion and optimizer
    model = VisionTransformerEncoderDecoder()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003)

    # Set up tokenizer
    digit_tokenizer = DigitTokenizer()

    # Training
    epochs = 3
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0
        for idx in range(len(train_dataset)):
            img, labels = train_dataset[idx]

            # Tokenize
            img_patches, patches_pos, img_height, img_width = Tokenize_multidigit_img(img)
            decoder_input = digit_tokenizer.create_decoder_input(labels)
            decoder_target = digit_tokenizer.create_decoder_target(labels)

            # Forward pass
            optimizer.zero_grad()
            output_logits = model(img_patches, decoder_input)
            loss = criterion(output_logits, decoder_target)
            epoch_train_loss += loss.item()
            
            # Backward pass and update weights
            loss.backward()
            optimizer.step()
              
            # Print stats every 100 batches
            if idx % 10000 == 0: 
                print(f'Train Epoch: {epoch} [{idx}/{len(train_dataset)} '
                      f'({100. * idx / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}'
                      f'Example of image patches that are inputted to encoder model: {img_patches}'
                      f'Example of decoder input: {decoder_input}'
                      f'Example of decoder target: {decoder_target}')
        
        avg_train_loss = epoch_train_loss / len(train_dataset)
        
        
        # Test model after each epoch and print stats
        model.eval()
        epoch_test_loss = 0
        correct = 0
        for idx in range(len(test_dataset)):
            img, labels = test_dataset[idx]

            # Tokenize
            img_patches, patches_pos, img_height, img_width = Tokenize_multidigit_img(img)
            
            decoder_input = digit_tokenizer.create_decoder_input(labels)
            decoder_target = digit_tokenizer.create_decoder_target(labels)

            # Forward pass
            output_logits = model(img_patches, decoder_input)
            loss = criterion(output_logits, decoder_target)
            epoch_test_loss += loss.item()
            pred = output_logits.argmax(dim=1)
            correct += (pred == decoder_target).sum().item()
        avg_test_loss = epoch_test_loss/len(test_dataset)
    
        print(f'Train set: Average loss: {avg_train_loss:.4f}, '
            f'Test set: Average loss: {avg_test_loss:.4f}, '
            f'Accuracy: {correct}/{(len(test_dataset)*17)} ' # 17 is the sequence lenth, ie for each example, there are 17 tokens to predict correctly
            f'({100. * correct / (len(test_dataset)*17):.2f}%)')
        
        # Save the trained model
        torch.save(model.state_dict(), f'models/encoder_decoder_model_{datetime.now()}.pt') #Saves the trained model's parameters (weights and biases) 

    
if __name__ == "__main__":
    train_multidigit_encoder_decoder_model()
    