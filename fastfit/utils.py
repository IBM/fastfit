from datasets import Dataset


def sample_dataset(
    dataset, label_column: str = "label", num_samples_per_label: int = 8, seed: int = 42
):
    """Samples a Dataset to create an equal number of samples per class (when possible)."""
    shuffled_dataset = dataset.shuffle(seed=seed)

    df = shuffled_dataset.to_pandas()
    df = df.groupby(label_column)

    # sample num_samples, or at least as much as possible
    df = df.apply(
        lambda x: x.sample(min(num_samples_per_label, len(x)), random_state=seed)
    )
    df = df.reset_index(drop=True)

    all_samples = Dataset.from_pandas(df, features=dataset.features)
    return all_samples.shuffle(seed=seed)
