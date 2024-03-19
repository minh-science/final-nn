# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    # Separate positive and negative sequences using list comprehension using zip to ensure each seq is not separated from label
    positive_seqs = [seq for seq, label in zip(seqs, labels) if label == 1 ] 
    negative_seqs = [seq for seq, label in zip(seqs, labels) if label == 0]

    # Determine the number of samples to select (minimum of both classes)
    num_samples = min(len(positive_seqs), len(negative_seqs))

    # Sample sequences from both classes equally and randomly  
    sampled_positive = np.random.choice(positive_seqs, num_samples, replace=True)
    sampled_negative = np.random.choice(negative_seqs, num_samples, replace=True)

    # Combine sampled sequences and labels
    sampled_seqs = list(sampled_positive) + list(sampled_negative)
    sampled_labels = [1] * num_samples + [0] * num_samples
    
    return sampled_seqs, sampled_labels



def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    encoding_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}

    encodings = np.ones( (len(seq_arr), 4 * len(seq_arr[0]) ))

    for i in range(len(seq_arr)):
        i_seq = seq_arr[i]
        i_encode = []
        for j in i_seq:
            i_encode.extend(encoding_dict[j])
        encodings[i] = i_encode
    return np.array(encodings)