import torch
import torch.nn.functional as F

__all__ = [
    "sampled_clients_identifier",
    "tensor_concater",
    "dict_concater",
]


def sampled_clients_identifier(data_distributed, sampled_clients):
    """Identify local datasets information (distribution, size)"""
    local_dist_list, local_size_list = [], []

    for client_idx in sampled_clients:
        local_dist = torch.Tensor(data_distributed["data_map"])[client_idx]
        local_dist = F.normalize(local_dist, dim=0, p=1)
        local_dist_list.append(local_dist.tolist())

        local_size = data_distributed["local"][client_idx]["datasize"]
        local_size_list.append(local_size)

    return local_dist_list, local_size_list


def tensor_concater(tensor1, tensor2, device=None):
    """Concatenate two tensors"""

    if tensor1 is None:
        tensor1 = tensor2

    else:
        if device is not None:
            tensor1 = tensor1.to(device)
            tensor2 = tensor2.to(device)

        tensor1 = torch.cat((tensor1, tensor2), dim=0)

    return tensor1.to(device)


def dict_concater(dict1, dict2):
    """Concatenate two dictionaries"""
    for key, item in dict2.items():
        dict1[key] = item

    return dict1


def calibration_logits(self, input_logits, distribution_maps):
    total_maps = sum(distribution_maps)
    if total_maps == 0:
        raise ValueError("The sum of the weight array must not be zero!!")
    normalized_maps = distribution_maps / total_maps
    maps_tensor = torch.from_numpy(normalized_maps).to(self.device)
    log_maps = maps_tensor.log().clamp_(min=-1e9)
    denominator = torch.logsumexp(log_maps + input_logits, dim=-1, keepdim=True)
    return log_maps + input_logits - denominator