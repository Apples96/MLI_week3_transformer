import torch
import torch.nn as nn
import torch.nn.functional as F


# ATTENTION CLASSES

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
    def forward (self, x):
         query_emb = self.Wq(x) # (5 * 64)
         key_emb = self.Wk(x) # (5 * 64)
         value_emb = self.Wv(x) # (5 * 64)
         # For debugging, print the shapes
         sims = query_emb @ key_emb.transpose(0,1) / (self.embed_dim ** 0.5) # (5 * 64) @ (64*5) > (5*5)
         scaled_sims = F.softmax(sims, dim = 1) # (5*5)
         x = scaled_sims @ value_emb # (5*5) @ (5*64) > (5*64)
         return x
    
class MaskedSelfAttention(nn.Module):
    def __init__(self, embed_dim, seq_length = 17):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
        self.mask = torch.tril(torch.ones(seq_length, seq_length)) # Lower triangular matrix of ones (including the diagonal)
    def forward (self, x):
         query_emb = self.Wq(x) # (17*64)
         key_emb = self.Wk(x) # (17*64)
         value_emb = self.Wv(x) # (17*64)
         # For debugging, print the shapes
         sims = query_emb @ key_emb.transpose(0,1) / (self.embed_dim ** 0.5) # ((17*64) @ (64*17) > (17*17)
         masked_sims = sims.masked_fill(self.mask == 0, -1e9)
         scaled_masked_sims = F.softmax(masked_sims, dim = 1) # (17*17)
         x = scaled_masked_sims @ value_emb # (17*17) @ (17*64) > (17*64)
         return x
    
class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim)
    def forward (self, encoder_hidden_state, decoder_hidden_state):
         decoder_query_emb = self.Wq(decoder_hidden_state) # (17 * 64)
         encoder_key_emb = self.Wk(encoder_hidden_state) # (65 * 64)
         encoder_value_emb = self.Wv(encoder_hidden_state) # (65 * 64)
         # For debugging, print the shapes
         sims = decoder_query_emb @ encoder_key_emb.transpose(0,1) / (self.embed_dim ** 0.5) # (17 * 64) @ (64*65) > (17*65)
         scaled_sims = F.softmax(sims, dim = 1) # (17*65)
         x = scaled_sims @ encoder_value_emb # (17*65) @ (65*64) > (17*64)
         return x

     


# MODEL CLASSES

class VisionTransformerEncoderDecoder(nn.Module):
    def __init__(self, 
                img_size=28,            # MNIST images are 28x28
                sequence_length = 17,
                vocab_size=12,          # Each patch is 14x14 (gives 4 patches per image)
                in_channels=1,          # MNIST is grayscale (1 channel)
                num_classes=10,         # 10 digits for classification
                embed_dim=64,           # Embedding dimension
                num_heads=1,            # Number of attention heads
                num_encoder_layers=6,   # Number of encoder transformer layers
                num_decoder_layers=6,   # Number of decoder transformer layers
                mlp_ratio=4.0,          # Ratio for MLP hidden dim
                dropout=0.1,            # Dropout rate
            ): 
        super().__init__()
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.embed_dim = embed_dim
        
        self.encoder_model = VisionTransformerEncoder(num_patches = 64,        # 16 patches per image (4 patches per base image, and 4 base images per image)
                img_size=112,            # MNIST images are 28x28
                patch_size=14,          # Each patch is 14x14 (gives 4 patches per image)
                in_channels=1,          # MNIST is grayscale (1 channel)
                embed_dim=64,          # Embedding dimension
                num_heads=1,            # Number of attention heads
                num_layers=6,           # Number of transformer layers
                mlp_ratio=4.0,          # Ratio for MLP hidden dim
                dropout=0.1,)

        torch.manual_seed(42)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.pos_embed = torch.randn(self.sequence_length, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.masked_attention = MaskedSelfAttention(self.embed_dim)
        self.cross_attention = CrossAttention(self.embed_dim)
        self.ff1 = nn.Linear(self.embed_dim, self.embed_dim) 
        self.ff2 = nn.Linear(self.embed_dim, self.vocab_size) 

    def forward(self, img_patches, labels): 
        # Get the encoder hidden states
        CLStoken_logits, encoder_hidden_state = self.encoder_model(img_patches)

        # Go through the decoder layers
        x = self.embedding(labels) # (17*12) @ (12*64) + (12*64) > (17*64) (12 is vocab_size 10 digits + SOS + EOS ; 17 is sequence length ie 4*4 digit tokens + 1 SOS/EOS token)
        x = x + self.pos_embed # (17*64) + (17*64) > (17*64)
        x = self.layer_norm(x) # (17*64)
             
        for i in range(self.num_decoder_layers):
            residuals1 = x # (17*64)
            x = self.masked_attention(x) # (17*64)
            x = x + residuals1 # (17*64)
            x = self.layer_norm(x) # (17*64)
            residuals2 = x # (17*64)

            x = self.cross_attention(encoder_hidden_state, x) # (17*64)
            x = x + residuals2 # (17*64)
            x = self.layer_norm(x) # (17*64)
            residuals3 = x # (17*64)

            x = self.ff1(x) # (17*64) @ (64*64) + (17*64) > (17*64)
            x = x + residuals3 # (17*64)
            x = self.layer_norm(x) # (17*64)
        output_logits = self.ff2(x)  # (17*64) @ (64*12) > (17*12)
        return output_logits # (17*12)




class VisionTransformerEncoder(nn.Module):
    def __init__(self, 
                num_patches = 5,        # 4 patches per image +1 "patch" for CLS token for encoder classifier only
                img_size=28,            # MNIST images are 28x28
                patch_size=14,          # Each patch is 14x14 (gives 4 patches per image)
                in_channels=1,          # MNIST is grayscale (1 channel)
                num_classes=10,         # 10 digits for classification
                embed_dim=64,          # Embedding dimension
                num_heads=1,            # Number of attention heads
                num_layers=6,           # Number of transformer layers
                mlp_ratio=4.0,          # Ratio for MLP hidden dim
                dropout=0.1,            # Dropout rate
            ): 
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.embedding = nn.Linear(self.patch_size**2, self.embed_dim) 
        torch.manual_seed(42)
        self.pos_embed = torch.randn(self.num_patches, self.embed_dim)
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.attention = SelfAttention(self.embed_dim)
        self.ff1 = nn.Linear(self.embed_dim, self.embed_dim) 
        self.ff2 = nn.Linear(self.embed_dim, self.num_classes)



    def forward(self, img_patches):
        x = self.embedding(img_patches) # (5*196) @ (196*64) + (5*64) > (5*64)
        x = x + self.pos_embed # (4*64) + (5*64) > (5*64)
        x = self.layer_norm(x)
        for i in range(self.num_layers):
                residuals1 = x
                x = self.attention(x) # (5*64)
                x = x + residuals1 # (5*64)
                x = self.layer_norm(x) # (5*64)
                residuals2 = x # (5*64)
                
                x = self.ff1(x) # (5*64) @ (64*64) + (5*64) > (5*64)
                x = x + residuals2 # (5*64)
                x = self.layer_norm(x)
        encoder_hidden_state = x 
        logits = self.ff2(x)  # (5*64) @ (64*10) > (5*10)
        CLStoken_logits = logits[0, :]
        return CLStoken_logits, encoder_hidden_state
                









