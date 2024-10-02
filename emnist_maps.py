chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
emnist_class_mapping = {char: idx for idx, char in enumerate(chars)}
emnist_class_mapping_reversed = {idx: char for char, idx in emnist_class_mapping.items()}
