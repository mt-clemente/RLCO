import torch

def calculate_mean_rank(tensor):
    id_counts = torch.zeros(50)
    id_index_sum = torch.zeros(50)

    for row in tensor:
        for index, id_ in enumerate(row):
            id_counts[id_] += 1
            id_index_sum[id_] += index

    mean_ranks = id_index_sum / id_counts
    return mean_ranks

# Example usage
tensor = torch.randint(0, 50, (32, 50))  # Example tensor with 32 rows, each a permutation of IDs 0-49
mean_ranks = calculate_mean_rank(tensor)
print(mean_ranks)