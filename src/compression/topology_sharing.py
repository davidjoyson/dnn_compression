def apply_topology_sharing(model):
    """
    Replace all dendritic neurons' branch weights with a shared template.
    """
    template_linear1 = model.layer1.neurons[0].branch_linear1.data.clone()
    template_linear2 = model.layer1.neurons[0].branch_linear2.data.clone()

    for neuron in model.layer1.neurons:
        neuron.branch_linear1.data = template_linear1.clone()
        neuron.branch_linear2.data = template_linear2.clone()
