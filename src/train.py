import argparse
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

from src.dataset import get_datasets, tf_dataset_to_jax
from src.model import CapsNet
from src.utils import margin_loss, reconstruction_loss

def create_train_state(rng, model, learning_rate, input_shape):
    """Creates initial `TrainState`."""
    dummy_input = jnp.ones(input_shape)
    dummy_labels = jnp.ones((input_shape[0], model.num_classes))
    params = model.init(rng, dummy_input, dummy_labels)['params']
    
    # Paper uses Adam optimizer
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, images, labels, alpha=0.0005):
    """Trains for a single step."""
    def loss_fn(params):
        # Forward pass (training mode, passing labels)
        lengths, reconstructions = state.apply_fn({'params': params}, images, labels=labels)
        
        # Calculate losses
        m_loss = margin_loss(labels, lengths)
        r_loss = reconstruction_loss(images, reconstructions)
        
        # Total loss
        total_loss = m_loss + alpha * r_loss
        return total_loss, (m_loss, r_loss, lengths)
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (total_loss, (m_loss, r_loss, lengths)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    
    # Calculate accuracy
    predictions = jnp.argmax(lengths, axis=-1)
    true_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)
    
    metrics = {
        'loss': total_loss,
        'margin_loss': m_loss,
        'recon_loss': r_loss,
        'accuracy': accuracy
    }
    return state, metrics

@jax.jit
def eval_step(state, images, labels, alpha=0.0005):
    """Evaluates for a single step."""
    # Forward pass (testing mode, no labels passed for masking)
    lengths, reconstructions = state.apply_fn({'params': state.params}, images, labels=None)
    
    m_loss = margin_loss(labels, lengths)
    r_loss = reconstruction_loss(images, reconstructions)
    total_loss = m_loss + alpha * r_loss
    
    predictions = jnp.argmax(lengths, axis=-1)
    true_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(predictions == true_labels)
    
    metrics = {
        'loss': total_loss,
        'margin_loss': m_loss,
        'recon_loss': r_loss,
        'accuracy': accuracy
    }
    return metrics, reconstructions

def plot_reconstructions(original_images, reconstructed_images, dataset_name, out_dir="out"):
    """Plots and saves original vs reconstructed images."""
    os.makedirs(out_dir, exist_ok=True)
    num_images_to_plot = min(10, original_images.shape[0])
    
    plt.figure(figsize=(num_images_to_plot * 2, 4))
    for i in range(num_images_to_plot):
        # Original Image
        plt.subplot(2, num_images_to_plot, i + 1)
        img = original_images[i]
        if dataset_name == 'mnist':
            plt.imshow(img.squeeze(), cmap='gray')
        else:
            plt.imshow(img)
        plt.title("Original")
        plt.axis('off')
        
        # Reconstructed Image
        plt.subplot(2, num_images_to_plot, num_images_to_plot + i + 1)
        recon_img = reconstructed_images[i]
        if dataset_name == 'mnist':
            plt.imshow(recon_img.squeeze(), cmap='gray')
        else:
            plt.imshow(recon_img)
        plt.title("Reconstructed")
        plt.axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"{dataset_name}_reconstructions.png")
    plt.savefig(save_path)
    print(f"Saved reconstructions plot to {save_path}")
    plt.close()

def main(args):
    # Setup JAX and random seed
    print(f"Running on: {jax.devices()}")
    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng = jax.random.split(rng)

    # Load data
    print(f"Loading {args.dataset} dataset...")
    train_ds, test_ds, info = get_datasets(args.dataset, args.batch_size)
    
    # Get shape from dataset info
    sample_shape = info.features['image'].shape
    input_shape = (args.batch_size,) + sample_shape
    print(f"Input shape: {input_shape}")
    
    # Initialize model and state
    model = CapsNet(num_classes=10, dataset_name=args.dataset)
    state = create_train_state(init_rng, model, args.learning_rate, input_shape)
    
    # Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_metrics = []
        for images, labels in tqdm(tf_dataset_to_jax(train_ds), desc=f"Epoch {epoch+1} Train", leave=False):
            state, metrics = train_step(state, images, labels)
            train_metrics.append(metrics)
        
        # Aggregate train metrics
        train_loss = jnp.mean(jnp.array([m['loss'] for m in train_metrics]))
        train_acc = jnp.mean(jnp.array([m['accuracy'] for m in train_metrics]))
        
        # Evaluate
        eval_metrics = []
        last_eval_batch_images = None
        last_eval_batch_recons = None
        for images, labels in tqdm(tf_dataset_to_jax(test_ds), desc=f"Epoch {epoch+1} Eval", leave=False):
            metrics, reconstructions = eval_step(state, images, labels)
            eval_metrics.append(metrics)
            # Keep the last batch for visualization
            last_eval_batch_images = images
            last_eval_batch_recons = reconstructions
            
        # Aggregate eval metrics
        eval_loss = jnp.mean(jnp.array([m['loss'] for m in eval_metrics]))
        eval_acc = jnp.mean(jnp.array([m['accuracy'] for m in eval_metrics]))
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} in {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        print(f"  Test Loss: {eval_loss:.4f}, Accuracy: {eval_acc:.4f}")
        
    # Plot reconstructions from the last evaluation batch
    if last_eval_batch_images is not None and last_eval_batch_recons is not None:
        print("Generating reconstruction plots...")
        plot_reconstructions(last_eval_batch_images, last_eval_batch_recons, args.dataset, args.out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CapsNet in JAX/Flax')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default='out')
    args = parser.parse_args()
    
    # Run from root dir so modules import correctly
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    main(args)
