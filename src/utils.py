import jax.numpy as jnp

def squash(x, axis=-1, epsilon=1e-7):
    """
    The non-linear activation function used in CapsNet.
    It ensures that the length of a vector is between 0 and 1.
    
    Args:
        x: Input tensor.
        axis: The axis along which to calculate the norm (usually the capsule dimension).
        epsilon: A small value to prevent division by zero during gradient calculation.
        
    Returns:
        The squashed tensor.
    """
    # Calculate the squared norm of the vector
    squared_norm = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    
    # Calculate the scale factor
    scale = squared_norm / (1.0 + squared_norm)
    
    # Calculate the unit vector. We add epsilon to the norm to prevent division by zero
    # where the norm is exactly 0.
    unit_vector = x / jnp.sqrt(squared_norm + epsilon)
    
    return scale * unit_vector

def margin_loss(labels, logits, m_plus=0.9, m_minus=0.1, lambda_val=0.5):
    """
    Calculates the margin loss for CapsNet.
    
    Args:
        labels: One-hot encoded true labels. Shape (batch_size, num_classes)
        logits: The lengths of the output capsules. Shape (batch_size, num_classes)
        m_plus: Margin for the true class.
        m_minus: Margin for the negative classes.
        lambda_val: Down-weighting factor for the negative classes to stop initial learning 
                    from shrinking the lengths of all capsules.
                    
    Returns:
        The mean margin loss over the batch.
    """
    # Loss for the correct class: max(0, m_plus - ||v_c||)^2
    present_error = jnp.square(jnp.maximum(0., m_plus - logits))
    
    # Loss for the incorrect classes: max(0, ||v_c|| - m_minus)^2
    absent_error = jnp.square(jnp.maximum(0., logits - m_minus))
    
    # Total loss: L_c = T_c * present_error + lambda * (1 - T_c) * absent_error
    loss = labels * present_error + lambda_val * (1.0 - labels) * absent_error
    
    # Sum over classes, mean over batch
    return jnp.mean(jnp.sum(loss, axis=-1))

def reconstruction_loss(images, reconstructions):
    """
    Calculates the reconstruction loss (MSE).
    
    Args:
        images: Original images (flattened or not).
        reconstructions: Reconstructed images.
        
    Returns:
        The MSE loss.
    """
    # Flatten images and reconstructions if they aren't already
    images_flat = jnp.reshape(images, (images.shape[0], -1))
    reconstructions_flat = jnp.reshape(reconstructions, (reconstructions.shape[0], -1))
    
    # Calculate Sum of Squared Differences (SSD) as in the paper
    # "We use the sum of squared differences between the outputs of the logistic units and the pixel intensities."
    ssd = jnp.sum(jnp.square(images_flat - reconstructions_flat), axis=-1)
    
    return jnp.mean(ssd)
