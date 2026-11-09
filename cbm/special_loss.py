from multiprocessing import Value

import torch
import torch.nn.functional as F 

def concept_loss(inputs: torch.Tensor, targets: torch.Tensor, concept_counts: torch.Tensor, reduction: str = "sum") -> torch.Tensor:

    # calculate weights from concept counts
    eps = 1e-8
    weights = (1.0 / concept_counts + eps) / len(concept_counts)

    # get "normal" cross-entropy loss
    bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = "none")

    # get weighted sum of loss with weights as above
    loss = bce_loss@weights

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(f"invalid reduction type: {reduction}")
    
    return loss


if __name__ == "__main__":
    g = torch.Generator(device = "cpu")
    inputs = torch.rand(size = (5, 3), generator = g, dtype = torch.float32)
    targets = torch.randint(low = 0, high = 2, size = (5, 3), generator = g, dtype = torch.float32)
    concept_counts = targets.sum(dim = 0)
    print(f"{concept_counts}")
    concept_loss = concept_loss(inputs, targets, concept_counts)

    print(f"{inputs = }")
    print(f"{targets = }")
    print(f"{concept_loss = }")