import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, Sequence, Any
from .utils import squash

class PrimaryCaps(nn.Module):
    channels: int
    capsule_dim: int
    kernel_size: Sequence[int]
    strides: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        assert self.channels % self.capsule_dim == 0, "Channels must be divisible by capsule_dim"
        
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='VALID',
            name='conv'
        )(x)
        
        batch_size = x.shape[0]
        x = jnp.reshape(x, (batch_size, -1, self.capsule_dim))
        return squash(x, axis=-1)

class DigitCaps(nn.Module):
    num_capsules: int = 10
    capsule_dim: int = 16
    routings: int = 3
    
    @nn.compact
    def __call__(self, x):
        batch_size, num_primary_caps, primary_caps_dim = x.shape
        W = self.param('W', nn.initializers.glorot_uniform(),
                       (num_primary_caps, self.num_capsules, primary_caps_dim, self.capsule_dim))
                       
        u_hat = jnp.einsum('bie,ijed->bijd', x, W)
        b = jnp.zeros((batch_size, num_primary_caps, self.num_capsules))
        
        for i in range(self.routings):
            c = jax.nn.softmax(b, axis=-1)
            s = jnp.sum(jnp.expand_dims(c, -1) * u_hat, axis=1)
            v = squash(s, axis=-1)
            
            if i < self.routings - 1:
                agreement = jnp.einsum('bijd,bjd->bij', u_hat, v)
                b = b + agreement
                
        return v

class Decoder(nn.Module):
    output_shape: Sequence[int]
    hidden1: int = 512
    hidden2: int = 1024
    
    @nn.compact
    def __call__(self, x):
        import math
        x = nn.Dense(features=self.hidden1)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden2)(x)
        x = nn.relu(x)
        
        flat_output_size = math.prod(self.output_shape)
        x = nn.Dense(features=flat_output_size)(x)
        x = nn.sigmoid(x)
        
        batch_size = x.shape[0]
        return jnp.reshape(x, (batch_size, *self.output_shape))

class CapsNet(nn.Module):
    num_classes: int = 10
    dataset_name: str = "mnist"
    conv_features: int = 256
    primary_channels: int = 256
    primary_dim: int = 8
    digit_dim: int = 16
    decoder_hidden1: int = 512
    decoder_hidden2: int = 1024
    
    @nn.compact
    def __call__(self, x, labels=None):
        batch_size = x.shape[0]
        original_shape = x.shape[1:]
        
        if self.dataset_name == "cifar10":
            conv1_stride = (1, 1) 
        else:
            conv1_stride = (1, 1)
            
        x = nn.Conv(features=self.conv_features, kernel_size=(9, 9), strides=conv1_stride, padding='VALID', name='conv1')(x)
        x = nn.relu(x)
        
        x = PrimaryCaps(channels=self.primary_channels, capsule_dim=self.primary_dim, kernel_size=(9, 9), strides=(2, 2), name='primary_caps')(x)
        
        capsules = DigitCaps(num_capsules=self.num_classes, capsule_dim=self.digit_dim, routings=3, name='digit_caps')(x)
        
        lengths = jnp.sqrt(jnp.sum(jnp.square(capsules), axis=-1) + 1e-7)
        
        if labels is not None:
            mask = labels
        else:
            predictions = jnp.argmax(lengths, axis=-1)
            mask = jax.nn.one_hot(predictions, self.num_classes)
            
        masked_capsules = capsules * jnp.expand_dims(mask, -1)
        decoder_input = jnp.reshape(masked_capsules, (batch_size, -1))
        
        reconstructions = Decoder(output_shape=original_shape, hidden1=self.decoder_hidden1, hidden2=self.decoder_hidden2, name='decoder')(decoder_input)
        
        return lengths, reconstructions
