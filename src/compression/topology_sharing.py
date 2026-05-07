def apply_topology_sharing(model):
    """
    Applies topology sharing by copying the weights of the first dendritic branch
    to all other branches. This reduces parameter count and enforces a shared
    computational topology across branches.

    Expected model attributes:
        model.branches : nn.ModuleList of Linear layers
    """

    # Ensure model has dendritic branches
    if not hasattr(model, "branches"):
        print("Warning: Model has no 'branches' attribute. Skipping topology sharing.")
        return

    branches = model.branches

    # If only one branch, nothing to share
    if branches is None or len(branches) <= 1:
        return

    # Use first branch as template
    base_w = branches[0].weight.data.clone()
    base_b = branches[0].bias.data.clone()

    # Copy weights to all other branches
    for b in branches[1:]:
        b.weight.data.copy_(base_w)
        b.bias.data.copy_(base_b)
