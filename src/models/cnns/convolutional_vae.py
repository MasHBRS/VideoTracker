import os
import numpy as np
import torch
import torch.nn as nn
"""
TODO:
    It only works on squared shape images, on decoder you may change it!
"""
class ConvolutionalVAE(nn.Module):
    def __init__(self, 
                 in_size=(1, 32, 32),  #(channels,width,height)
                 channel_sizes=[1,16,32,64,128],
                 latent_dim=64,
                 act_final="Sigmoid",
                 act_hidden="ReLU", 
                 kernel_size=3, 
                 padding=1,
                 stride=1):
        
        super().__init__()
        self.in_size = in_size
        self.channel_sizes = channel_sizes
        self.activation_hidden = self.get_activation(act_hidden)
        self.activation_final= self.get_activation(act_final)
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.latent_dim = latent_dim

        self.encoder,self.latent_input_size,self.before_z_image_dim = self._make_encoder()
        self.decoder = self._make_decoder()
        self.fc_mu = nn.Linear(self.latent_input_size, latent_dim)
        self.fc_sigma = nn.Linear(self.latent_input_size, latent_dim)
        self.latent_to_decoder=nn.Sequential (nn.Linear(latent_dim,self.latent_input_size))
        return
        
    def _make_encoder(self):
        """ Defining encoder """
        layers = []
        for i in range(len(self.channel_sizes)-1):
            layers.append( nn.Conv2d(in_channels=self.channel_sizes[i], 
                                     out_channels=self.channel_sizes[i+1], 
                                     kernel_size=self.kernel_size, 
                                     padding=self.padding,
                                     stride=self.stride) )
            layers.append(nn.BatchNorm2d(self.channel_sizes[i+1]))
            layers.append(self.activation_hidden)
            if i ==0:
                img_size = self.compute_image_size(list(self.in_size[1:3]), np.ones(2)*self.kernel_size, np.ones(2)*self.padding, np.ones(2)*self.stride)
            else:
                img_size = self.compute_image_size(img_size, np.ones(2)*self.kernel_size, np.ones(2)*self.padding, np.ones(2)*self.stride)
        layers.append(nn.Flatten())
        encoder = nn.Sequential(*layers)
        latent_input_size=int(np.prod(img_size)*self.channel_sizes[-1])
        return encoder, latent_input_size,img_size
    
    def _make_decoder(self):
        """ Defining decoder """
        layers = []
        
        for i in range(1, len(self.channel_sizes)):
            layers.append( nn.ConvTranspose2d(in_channels=self.channel_sizes[-i], 
                                              out_channels=self.channel_sizes[-i-1], 
                                              output_padding=1,
                                              kernel_size=self.kernel_size, 
                                              padding=self.padding,
                                              stride=self.stride) )
            layers.append(nn.BatchNorm2d(self.channel_sizes[-i-1]))
            layers.append(self.activation_hidden)

        layers = layers[:-2] + [self.activation_final]
        decoder = nn.Sequential(*layers)
        return decoder
    
    def reparameterize(self, mu, log_var):
        """ Reparametrization trick"""
        std = torch.exp(0.5*log_var)  # we can also predict the std directly, but this works best
        eps = torch.randn_like(std)  # random sampling happens here
        z = mu + std * eps
        return z
    def get_image_dimensions_for_decoder(self,z_to_decoder):
        z_to_decoder = z_to_decoder.view(-1, self.channel_sizes[-1], int(self.before_z_image_dim[-1]), int(self.before_z_image_dim[-2]))
        return z_to_decoder


    def forward(self, x):
        """ Forward pass """
        x_enc = self.encoder(x)
        mu = self.fc_mu(x_enc)
        log_var = self.fc_sigma(x_enc)
        z = self.reparameterize(mu, log_var)
        z_to_decoder=self.latent_to_decoder(z)
        z_to_decoder=self.get_image_dimensions_for_decoder(z_to_decoder)
        x_hat_flat = self.decoder(z_to_decoder)
        x_hat = x_hat_flat.view(-1, *self.in_size)
        return x_hat, (z, mu, log_var)
    
    def get_activation(self,act_name):
        """ Gettign activation given name """
        assert act_name in ["ReLU", "Sigmoid", "Tanh"]
        activation = getattr(nn, act_name)
        return activation()
    
    def compute_image_size(img_size, kernel_size, padding, stride):
        """
        Compute the output size of a convolutional layer given the input size, kernel size, padding, and stride.
        """
        return (img_size - kernel_size + 2 * padding) // stride + 1
