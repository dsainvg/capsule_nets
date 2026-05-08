import json
import os

utils_code = """
import jax.numpy as jnp

def squash(x, axis=-1, epsilon=1e-7):
    squared_norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    scale = squared_norm / (1.0 + squared_norm)
    unit_vector = x / jnp.sqrt(squared_norm + epsilon)
    return scale * unit_vector

def margin_loss(labels, logits, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
    present_error = jnp.square(jnp.maximum(0., m_plus - logits))
    absent_error = jnp.square(jnp.maximum(0., logits - m_minus))
    loss = labels * present_error + lambda_val * (1.0 - labels) * absent_error
    return jnp.mean(jnp.sum(loss, axis=-1))

def reconstruction_loss(images, reconstructions):
    images_flat = jnp.reshape(images, (images.shape[0], -1))
    reconstructions_flat = jnp.reshape(reconstructions, (reconstructions.shape[0], -1))
    ssd = jnp.sum(jnp.square(images_flat - reconstructions_flat), axis=-1)
    return jnp.mean(ssd)
"""

model_code = """
import jax
import flax.linen as nn
from typing import Callable, Sequence, Any

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
"""

dataset_code = """
import tensorflow as tf

class DummyInfo:
    def __init__(self, num_classes, image_shape):
        self.features = {'image': DummyImageFeature(image_shape), 'label': DummyLabelFeature(num_classes)}

class DummyImageFeature:
    def __init__(self, shape):
        self.shape = shape

class DummyLabelFeature:
    def __init__(self, num_classes):
        self.num_classes = num_classes

def get_datasets(dataset_name="mnist", batch_size=128, subset_size=None):
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train[..., tf.newaxis]
        x_test = x_test[..., tf.newaxis]
        num_classes = 10
        image_shape = (28, 28, 1)
    elif dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        num_classes = 10
        image_shape = (32, 32, 3)
    else:
        raise ValueError("Unknown dataset")
        
    info = DummyInfo(num_classes, image_shape)

    def preprocess(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        label = tf.one_hot(label, num_classes)
        return image, label
        
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if subset_size:
        train_ds = train_ds.take(subset_size)
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    if subset_size:
        test_ds = test_ds.take(max(100, subset_size // 5))
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, info

def tf_dataset_to_jax(dataset):
    for batch in dataset:
        images, labels = batch
        yield jnp.array(images.numpy()), jnp.array(labels.numpy())
"""

train_setup_code = """
# Kaggle Run Configuration
DATASET = 'mnist'
EPOCHS = 7
LEARNING_RATE = 0.001
SUBSET_SIZE = None  # None uses the full dataset

# Multi-GPU Setup
num_devices = jax.local_device_count()
print(f"Number of JAX devices available: {num_devices}")

# Global batch size must be divisible by the number of devices
# We use 128 images per GPU
BATCH_SIZE = 128 * num_devices

# Scaled up model parameters
CONV_FEATURES = 512       
PRIMARY_CHANNELS = 512    
PRIMARY_DIM = 16          
DIGIT_DIM = 32            
DECODER_HIDDEN1 = 1024    
DECODER_HIDDEN2 = 2048    

def shard_batch(batch):
    images, labels = batch
    # Reshape to (num_devices, batch_size_per_device, ...)
    images = jnp.reshape(images, (num_devices, -1) + images.shape[1:])
    labels = jnp.reshape(labels, (num_devices, -1) + labels.shape[1:])
    return images, labels

def create_train_state(rng, model, learning_rate, input_shape):
    dummy_input = jnp.ones(input_shape)
    dummy_labels = jnp.ones((input_shape[0], model.num_classes))
    params = model.init(rng, dummy_input, dummy_labels)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# jax.pmap applies the function across all available devices
@functools.partial(jax.pmap, axis_name='batch')
def p_train_step(state, images, labels, alpha=0.0005):
    def loss_fn(params):
        lengths, reconstructions = state.apply_fn({'params': params}, images, labels=labels)
        m_loss = margin_loss(labels, lengths)
        r_loss = reconstruction_loss(images, reconstructions)
        total_loss = m_loss + alpha * r_loss
        return total_loss, (m_loss, r_loss, lengths)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (m_loss, r_loss, lengths)), grads = grad_fn(state.params)
    
    # Average gradients across all devices before updating
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    
    predictions = jnp.argmax(lengths, axis=-1)
    true_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)
    
    metrics = {'loss': total_loss, 'margin_loss': m_loss, 'accuracy': accuracy}
    # Average metrics across all devices
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    return state, metrics

@functools.partial(jax.pmap, axis_name='batch')
def p_eval_step(state, images, labels, alpha=0.0005):
    lengths, reconstructions = state.apply_fn({'params': state.params}, images, labels=None)
    m_loss = margin_loss(labels, lengths)
    r_loss = reconstruction_loss(images, reconstructions)
    total_loss = m_loss + alpha * r_loss
    
    predictions = jnp.argmax(lengths, axis=-1)
    true_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)
    
    metrics = {'loss': total_loss, 'margin_loss': m_loss, 'accuracy': accuracy}
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    return metrics, reconstructions
"""

loop_code = """
print(f"Loading {DATASET} dataset (Subset size: {SUBSET_SIZE if SUBSET_SIZE else 'Full'})...")
train_ds, test_ds, info = get_datasets(DATASET, BATCH_SIZE, subset_size=SUBSET_SIZE)

# We use batch size of 1 for the dummy shape during model initialization on the host
dummy_input_shape = (1,) + info.features['image'].shape

model = CapsNet(
    num_classes=10,
    dataset_name=DATASET,
    conv_features=CONV_FEATURES,
    primary_channels=PRIMARY_CHANNELS,
    primary_dim=PRIMARY_DIM,
    digit_dim=DIGIT_DIM,
    decoder_hidden1=DECODER_HIDDEN1,
    decoder_hidden2=DECODER_HIDDEN2
)

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)

# Create state on host
state = create_train_state(init_rng, model, LEARNING_RATE, dummy_input_shape)

# Replicate the state to all devices
state = jax_utils.replicate(state)

for epoch in range(EPOCHS):
    start_time = time.time()
    
    # Train
    train_metrics = []
    for batch in tqdm(tf_dataset_to_jax(train_ds), desc=f"Epoch {epoch+1} Train", leave=False):
        images, labels = shard_batch(batch)
        state, metrics = p_train_step(state, images, labels)
        
        # metrics are identical across devices because of pmean, so we just take the 0th device's copy
        train_metrics.append({k: v[0] for k, v in metrics.items()})
    
    train_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics]))
    train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
    
    # Evaluate
    eval_metrics = []
    last_images, last_recons = None, None
    for batch in tqdm(tf_dataset_to_jax(test_ds), desc=f"Epoch {epoch+1} Eval", leave=False):
        images, labels = shard_batch(batch)
        metrics, reconstructions = p_eval_step(state, images, labels)
        
        eval_metrics.append({k: v[0] for k, v in metrics.items()})
        
        # Save one device's batch for plotting
        last_images, last_recons = images[0], reconstructions[0]
        
    eval_loss = jnp.mean(jnp.array([m['loss'] for m in eval_metrics]))
    eval_acc = jnp.mean(jnp.array([m['accuracy'] for m in eval_metrics]))
    
    print(f"Epoch {epoch+1} in {time.time() - start_time:.2f}s")
    print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"  Test Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
"""

plot_code = """
if last_images is not None:
    num_plot = min(10, last_images.shape[0])
    plt.figure(figsize=(num_plot * 2, 4))
    for i in range(num_plot):
        # Original
        plt.subplot(2, num_plot, i + 1)
        plt.imshow(last_images[i].squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed
        plt.subplot(2, num_plot, num_plot + i + 1)
        plt.imshow(last_recons[i].squeeze(), cmap='gray')
        plt.title("Recon")
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
"""

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# CapsNet on Kaggle (Standalone & Multi-GPU Version)\n",
    "This notebook trains a scaled-up version of CapsNet on the full MNIST dataset. It contains all the necessary source code within the notebook itself, so you don't need to clone any external repositories. It also automatically detects and utilizes multiple GPUs using JAX `pmap` and `replicate`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import sys\n",
    "import functools\n",
    "import jax\n",
    "import jax.numpy as jnp\n",
    "import optax\n",
    "from flax.training import train_state\n",
    "from flax import jax_utils\n",
    "import time\n",
    "from tqdm.auto import tqdm\n",
    "import matplotlib.pyplot as plt\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 1. Utils & Loss Functions"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [utils_code.strip() + "\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 2. CapsNet Model Architecture"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [model_code.strip() + "\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 3. Dataset Pipeline"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [dataset_code.strip() + "\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 4. Kaggle Configuration & Setup (Multi-GPU)"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [train_setup_code.strip() + "\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 5. Training Loop"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [loop_code.strip() + "\n"]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["## 6. Plot Reconstructions"]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [plot_code.strip() + "\n"]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(os.path.join(r"R:\Reasearch Project\capsule_nets", "kaggle_run.ipynb"), "w") as f:
    json.dump(notebook, f, indent=1)
