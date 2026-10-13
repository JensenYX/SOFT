import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, hidden_size, rank, cap, init_std=1e-4):
        super().__init__()
        if hidden_size <= 0 or rank <= 0 or cap <= 0:
            raise ValueError("hidden_size, rank and cap must be positive")
        self.cap = cap
        self.down = nn.Parameter(torch.empty(hidden_size, rank))
        self.up = nn.Parameter(torch.zeros(rank, hidden_size))
        nn.init.normal_(self.down, std=init_std)

    def forward(self, hidden):
        delta = hidden.float() @ self.down @ self.up
        scale = (self.cap / delta.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        return hidden + (delta * scale).to(hidden.dtype)


class FastWeights(nn.Module):
    def __init__(self, hidden_size, rank, layers, cap):
        super().__init__()
        self.layers = list(layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("layers must be nonempty and unique")
        self.adapters = nn.ModuleDict({
            str(layer): ResidualAdapter(hidden_size, rank, cap)
            for layer in self.layers
        })

    def decay_(self, factor):
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.mul_(factor)

    @property
    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def resolve_module(module, path):
    for name in path.split("."):
        module = getattr(module, name)
    return module


def install_hooks(model, fast_weights, layer_path):
    layer_stack = resolve_module(model, layer_path)
    handles = []
    for layer in fast_weights.layers:
        adapter = fast_weights.adapters[str(layer)]

        def apply_adapter(_, __, output, adapter=adapter):
            if isinstance(output, tuple):
                return (adapter(output[0]),) + output[1:]
            if isinstance(output, list):
                return [adapter(output[0]), *output[1:]]
            return adapter(output)

        handles.append(layer_stack[layer].register_forward_hook(apply_adapter))
    return handles


def remove_hooks(handles):
    for handle in handles:
        handle.remove()
