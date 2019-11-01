# !pip install nltk
import nltk
nltk.download('punkt')

import torch
import torch.nn as nn
from torchvision import transforms
import sys
sys.path.append('/opt/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from data_loader import get_loader
from model import EncoderCNN, DecoderRNN
import math
from torch import optim


## TODO #1: Select appropriate values for the Python variables below.
batch_size = 12          # batch size
vocab_threshold = 5        # minimum word count threshold
vocab_from_file = True    # if True, load existing vocab file
embed_size = 512           # dimensionality of image and word embeddings
hidden_size = 256          # number of features in hidden state of the RNN decoder
num_epochs = 3             # number of training epochs
save_every = 1             # determines frequency of saving model weights
print_every = 100          # determines window for printing average loss
log_file = 'training_log.txt'       # name of file with saved training loss and perplexity

# (Optional) TODO #2: Amend the image transform below.
transform_train = transforms.Compose([ 
    transforms.Resize(256),                          # smaller edge of image resized to 256
    transforms.RandomAffine(degrees=15, translate=(.1,.1), shear=(10,10)),
    transforms.RandomCrop(224),                      # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),               # horizontally flip image with probability=0.5
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, 
                           hue=0.1),  # randomly change the brightness, contrast and saturation of an image
    transforms.ToTensor(),                           # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),      # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])

# Build data loader.
data_loader = get_loader(transform=transform_train,
                         mode='train',
                         batch_size=batch_size,
                         vocab_threshold=vocab_threshold,
                         vocab_from_file=vocab_from_file,
                         cocoapi_loc='/home/jupyter/')

# The size of the vocabulary.
vocab_size = len(data_loader.dataset.vocab)

from torch.optim.lr_scheduler import StepLR

# Initialize the encoder and decoder. 
encoder = EncoderCNN(embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

# Move models to GPU if CUDA is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
decoder.to(device)

# Define the loss function. 
criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()

# TODO #3: Specify the learnable parameters of the model.
# train all parameters except resnet weights...
# list(filter(lambda p: p.requires_grad, decoder.parameters()))
params = list(decoder.parameters()) + list(encoder.embed.parameters()) 

# TODO #4: Define the optimizer.
lr1=0.001
# lr2=0.0001
optimizer = optim.Adam(params, lr=lr1)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

# Set the total number of training steps per epoch.
total_step = math.ceil(len(data_loader.dataset.caption_lengths) / data_loader.batch_sampler.batch_size)

import time
t = round(time.time())
log_file = 'training_log_'+str(t)+'.txt'

import torch.utils.data as data
import numpy as np
import os
# import requests
import time

# Open the training log file.
f = open(log_file, 'w')

# old_time = time.time()
# response = requests.request("GET", 
#                             "http://metadata.google.internal/computeMetadata/v1/instance/attributes/keep_alive_token", 
#                             headers={"Metadata-Flavor":"Google"})

output_path = './models'

if not os.path.exists(output_path):
    os.makedirs(output_path)

for epoch in range(1, num_epochs+1):
    
    for i_step in range(1, total_step+1):
        
#         if time.time() - old_time > 60:
#             old_time = time.time()
#             requests.request("POST", 
#                              "https://nebula.udacity.com/api/v1/remote/keep-alive", 
#                              headers={'Authorization': "STAR " + response.text})
        
        # Randomly sample a caption length, and sample indices with that length.
        indices = data_loader.dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader.batch_sampler.sampler = new_sampler
        
        # Obtain the batch.
        images, captions = next(iter(data_loader))

        # Move batch of images and captions to GPU if CUDA is available.
        images = images.to(device)
        captions = captions.to(device)
        
        # Zero the gradients.
        decoder.zero_grad()
        encoder.zero_grad()
        
        # Pass the inputs through the CNN-RNN model.
        features = encoder(images)
        outputs = decoder(features, captions)
        
        # Calculate the batch loss.
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backward pass.
        loss.backward()
        
        # Update the parameters in the optimizer.
        optimizer.step()
        
        # Get training statistics.
        stats = 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f' % (epoch, num_epochs, i_step, total_step, loss.item(), np.exp(loss.item()))
        
        # Print training statistics (on same line).
        print('\r' + stats, end="")
        sys.stdout.flush()
        
        # Print training statistics to file.
        f.write(stats + '\n')
        f.flush()
        
        # Print training statistics (on different line).
        if i_step % print_every == 0:
            print('\r' + stats)
        
#         if epoch == 3 and i_step % 5000 == 0:
#             torch.save(decoder.state_dict(), os.path.join('./models', 'decoder-%d-%d.pkl' % epoch, i_step))
#             torch.save(encoder.state_dict(), os.path.join('./models', 'encoder-%d-%d.pkl' % epoch, i_step))
    # Save the weights.
    if epoch % save_every == 0:
        torch.save(decoder.state_dict(), os.path.join(output_path, 'decoder-%d.pkl' % epoch))
        torch.save(encoder.state_dict(), os.path.join(output_path, 'encoder-%d.pkl' % epoch))
    
    scheduler.step()
        
# Close the training log file.
f.close()