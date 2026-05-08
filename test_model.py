import jax
import jax.numpy as jnp
from src.model import CapsNet

def test_model():
    model = CapsNet(num_classes=10, dataset_name="mnist")
    
    # Batch of 2 MNIST images
    images = jnp.ones((2, 28, 28, 1))
    
    # Initialize
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, images)
    
    # Run
    lengths, reconstructions = model.apply(variables, images)
    
    print("Test passed!")
    print("Lengths shape:", lengths.shape)
    print("Reconstructions shape:", reconstructions.shape)

if __name__ == "__main__":
    test_model()
