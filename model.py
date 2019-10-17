# models.py

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F



class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, pretrained_embeds=None, debug=False):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        if pretrained_embeds is not None:
            # torch.from_numpy(pretrained_embeds)
            self.word_embeddings = self.word_embeddings.weight.data.copy_(pretrained_embeds)
            self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dec = nn.Linear(hidden_size, vocab_size)
        
        self.debug = debug
    
    def forward(self, features, captions):
        if self.debug:
            print('features: ', features.shape)
            print('captions: ', captions.shape)
        
        embeds = self.word_embeddings(captions[:,:-1])
        if self.debug:
            print('embeds: ', embeds.shape)
        
        features = features.unsqueeze(1)
        if self.debug:
            print('unsq feats: ', features.shape)
        
        inputs = (features, embeds)
        inputs = torch.cat(inputs,1)
        if self.debug:
            print('inputs: ', inputs.shape)
        
#         inputs = inputs.view(len(inputs), 1, -1)
#         print('inputs.view', inputs.shape)
        
        h_shape = (1, inputs.shape[0], self.hidden_size)
        if self.debug:
            print('h-shape: ', h_shape)
        
        hidden = (torch.randn(h_shape).cuda(), 
                  torch.randn(h_shape).cuda())  # clean out hidden state
        
        out, hidden = self.lstm(inputs, hidden)
        if self.debug:
            print('out: ', out.shape)
            print('hidden: ', hidden[0].shape)
        
        token_space = self.dec(out)
        if self.debug:
            print('token_space: ', token_space.shape)
        
#         token_scores = F.log_softmax(x, dim=1) # nn.CrossEntory takes care of this
#         if self.debug:
#             print('token_scores: ', token_scores.shape)
        
        return token_space

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        # clear hidden state
        h_shape = (1, inputs.shape[0], self.hidden_size)
        hidden = (torch.randn(h_shape).cuda(), torch.randn(h_shape).cuda())
        # loop until <end> token is sampled or maximum length is reached
#         out, hidden = self.lstm()
        for i in range(max_len):
            out, hidden = self.lstm(inputs, hidden)
            if self.debug:
                print('out: ', out.shape)
            
            x = self.dec(out)
            if self.debug:
                print('x: ', x.shape)
            
            values, tokens = torch.max(x, dim=-1)
            if self.debug:
                print('tokens: ',tokens)
            token = tokens.item()
            output.append(token)
            if token==1:
                break
            inputs = self.word_embeddings(tokens)
        return output