import tensorflow as tf
import jax.numpy as jnp

class DummyInfo:
    def __init__(self, num_classes, image_shape):
        self.features = {'image': DummyImageFeature(image_shape), 'label': DummyLabelFeature(num_classes)}

class DummyImageFeature:
    def __init__(self, shape):
        self.shape = shape

class DummyLabelFeature:
    def __init__(self, num_classes):
        self.num_classes = num_classes

def get_datasets(dataset_name="mnist", batch_size=128):
    """
    Loads and preprocesses the specified dataset using tf.keras.datasets.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        batch_size: Batch size for training and evaluation.
        
    Returns:
        train_ds: Training dataset (tf.data.Dataset)
        test_ds: Evaluation dataset (tf.data.Dataset)
        info: Dataset info
    """
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
        # Normalize images to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # One-hot encode labels
        label = tf.one_hot(label, num_classes)
        return image, label
        
    # Prepare training dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.cache()
    train_ds = train_ds.shuffle(buffer_size=10000)
    train_ds = train_ds.batch(batch_size, drop_remainder=True)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    
    # Prepare test dataset
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size, drop_remainder=True)
    test_ds = test_ds.cache()
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, test_ds, info

def tf_dataset_to_jax(dataset):
    """
    Converts a tf.data.Dataset to a generator of JAX arrays.
    """
    for batch in dataset:
        images, labels = batch
        # Convert to numpy arrays, which JAX can consume directly
        yield jnp.array(images.numpy()), jnp.array(labels.numpy())
