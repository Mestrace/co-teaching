def get(data_name: str):
    """
    Returns:
        train_dataset : tf.data.Dataset <= (train_X, noisy, train_label_y)
        test_dataset : tf.data.Dataset <= (test_X, test_label_y)

    Usage:
        cifar10_noisy_train, cifar10_test = data.get("cifar10")
    """
    pass


def save(dataset, loc="./saved_data"):
    """Save dataset to a specific location
    """
    pass


def load_saved(dataset_name: str, loc="./saved_data"):
    """Load data from specific location
    """
    pass
