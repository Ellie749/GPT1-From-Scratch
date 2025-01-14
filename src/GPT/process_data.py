import torch


def train_validation_split(data, split=0.9):
    train_length = int(0.9 * len(data))
    train_dataset = data[:train_length]
    validation_dataset = data[train_length:]
    
    return train_dataset, validation_dataset


def make_LM_dataset(data: torch.tensor, sequence_length: int, batch_size: int) -> torch.tensor:
    
    total_size = len(data)
    usable_size = (total_size // (batch_size * sequence_length)) * (batch_size * sequence_length)
    trimmed_data = data[:usable_size+1]
    x = trimmed_data[:-1]
    y = trimmed_data[1:]

    return torch.tensor(x).reshape(-1, batch_size, sequence_length), torch.tensor(y).reshape(-1, batch_size, sequence_length)




