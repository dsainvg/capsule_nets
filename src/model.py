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
        # Apply convolution
        # We need `channels` total output channels, which will be reshaped into capsules.
        # Ensure channels is divisible by capsule_dim
        assert self.channels % self.capsule_dim == 0, "Channels must be divisible by capsule_dim"
        
        x = nn.Conv(
            features=self.channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding='VALID',
            name='conv'
        )(x)
        
        # Reshape to [batch, num_capsules, capsule_dim]
        # x shape is [batch, H, W, channels]
        batch_size = x.shape[0]
        
        # Calculate the number of capsules (H * W * (channels // capsule_dim))
        x = jnp.reshape(x, (batch_size, -1, self.capsule_dim))
        
        # Apply squash activation along the capsule dimension
        return squash(x, axis=-1)

class DigitCaps(nn.Module):
    num_capsules: int = 10
    capsule_dim: int = 16
    routings: int = 3
    
    @nn.compact
    def __call__(self, x):
        # x is the output of PrimaryCaps: [batch_size, num_primary_caps, primary_caps_dim]
        batch_size, num_primary_caps, primary_caps_dim = x.shape
        
        # W shape: [num_primary_caps, num_capsules, primary_caps_dim, capsule_dim]
        W = self.param('W', nn.initializers.glorot_uniform(),
                       (num_primary_caps, self.num_capsules, primary_caps_dim, self.capsule_dim))
                       
        # Compute the prediction vectors (u_hat)
        # u_hat shape: [batch_size, num_primary_caps, num_capsules, capsule_dim]
        # jnp.einsum is very efficient for this
        u_hat = jnp.einsum('bie,ijed->bijd', x, W)
        
        # Dynamic routing
        # Initial logits for routing coefficients, shape [batch_size, num_primary_caps, num_capsules]
        b = jnp.zeros((batch_size, num_primary_caps, self.num_capsules))
        
        # We use a python loop because routings is a small constant (e.g., 3).
        # JAX will unroll this loop during compilation.
        for i in range(self.routings):
            # Compute routing probabilities c_ij
            c = jax.nn.softmax(b, axis=-1)
            
            # Compute s_j = sum_i(c_ij * u_hat_j|i)
            # c shape: [batch_size, num_primary_caps, num_capsules]
            # expand c to [batch_size, num_primary_caps, num_capsules, 1] for broadcasting
            s = jnp.sum(jnp.expand_dims(c, -1) * u_hat, axis=1)
            
            # Apply squash to get v_j
            v = squash(s, axis=-1)
            
            # If not the last iteration, update the routing logits
            if i < self.routings - 1:
                # v shape: [batch_size, num_capsules, capsule_dim]
                # Expand v for dot product with u_hat
                # dot product between u_hat (bijd) and v (bjd) along dimension d
                agreement = jnp.einsum('bijd,bjd->bij', u_hat, v)
                b = b + agreement
                
        return v

class Decoder(nn.Module):
    output_shape: Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        import math
        # x is the active digit capsule or all digit capsules masked
        # Shape: [batch_size, num_capsules * capsule_dim]
        x = nn.Dense(features=1024)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2048)(x)
        x = nn.relu(x)
        
        flat_output_size = math.prod(self.output_shape)
        x = nn.Dense(features=flat_output_size)(x)
        x = nn.sigmoid(x)
        
        # Reshape to the original image shape
        batch_size = x.shape[0]
        return jnp.reshape(x, (batch_size, *self.output_shape))

class CapsNet(nn.Module):
    num_classes: int = 10
    dataset_name: str = "mnist" # Used to adjust architecture slightly for cifar
    
    @nn.compact
    def __call__(self, x, labels=None):
        batch_size = x.shape[0]
        original_shape = x.shape[1:]
        
        # Adjust initial convolution based on dataset to manage memory/receptive field
        if self.dataset_name == "cifar10":
            # CIFAR is 32x32x3. Use a stride of 2 to reduce spatial dimensions early
            # or keep it 1 and rely on PrimaryCaps stride. Let's stick closer to original 
            # but maybe reduce filters to fit in memory if needed. We'll use stride 1 for now.
            conv1_stride = (1, 1) 
        else:
            conv1_stride = (1, 1)
            
        # 1. Conv1
        x = nn.Conv(features=512, kernel_size=(9, 9), strides=conv1_stride, padding='VALID', name='conv1')(x)
        x = nn.relu(x)
        
        # 2. PrimaryCaps
        x = PrimaryCaps(channels=512, capsule_dim=16, kernel_size=(9, 9), strides=(2, 2), name='primary_caps')(x)
        
        # 3. DigitCaps
        capsules = DigitCaps(num_capsules=self.num_classes, capsule_dim=32, routings=3, name='digit_caps')(x)
        
        # Calculate lengths of capsules for classification prediction
        # capsules shape: [batch_size, num_classes, 16]
        lengths = jnp.sqrt(jnp.sum(jnp.square(capsules), axis=-1) + 1e-7) # Add epsilon for stability
        
        # Reconstruction Masking
        # During training, we mask out all capsules except the one corresponding to the true label.
        # During testing, we mask out all capsules except the one with the longest length.
        if labels is not None:
            # Training phase
            mask = labels
        else:
            # Testing phase: create one-hot mask based on longest capsule
            predictions = jnp.argmax(lengths, axis=-1)
            mask = jax.nn.one_hot(predictions, self.num_classes)
            
        # Apply mask
        # Expand mask to [batch_size, num_classes, 1] and multiply
        masked_capsules = capsules * jnp.expand_dims(mask, -1)
        
        # Flatten for the decoder
        decoder_input = jnp.reshape(masked_capsules, (batch_size, -1))
        
        # 4. Decoder
        reconstructions = Decoder(output_shape=original_shape, name='decoder')(decoder_input)
        
        return lengths, reconstructions
