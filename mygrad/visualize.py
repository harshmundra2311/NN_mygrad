from graphviz import graphs
from graphviz import Digraph
# code to visualize the graphs
from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir' : 'LR'})

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{%s | data %.4f | grad %.4f}" % (n.label, n.data, n.grad), shape="record")
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid+n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    return dot


def draw_mlp(mlp, nin):
    """
    Draws a macroscopic view of the MLP.
    Blue edges are positive weights, Red are negative weights.
    Edge thickness represents the magnitude of the weight.
    """
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR', 'splines': 'line'})
    
    # 1. Draw the Input Nodes
    with dot.subgraph(name='cluster_input') as c:
        c.attr(color='white') # Hide the bounding box
        for i in range(nin):
            c.node(f'in_{i}', label=f'Input {i}', shape='circle', style='filled', fillcolor='lightgrey')
            
    # 2. Draw the Hidden and Output Nodes
    for l_idx, layer in enumerate(mlp.layers):
        with dot.subgraph(name=f'cluster_{l_idx}') as c:
            c.attr(color='white')
            for n_idx, neuron in enumerate(layer.neurons):
                # The label inside the neuron is its Bias
                b = neuron.b.data
                c.node(f'L{l_idx}_{n_idx}', label=f'b={b:.2f}', shape='circle')
                
    # 3. Draw the Edges from Inputs -> First Layer
    for i in range(nin):
        for n_idx, neuron in enumerate(mlp.layers[0].neurons):
            w = neuron.w[i].data
            color = 'blue' if w > 0 else 'red'
            penwidth = str(max(0.5, abs(w) * 3)) # Thicker lines for bigger weights
            dot.edge(f'in_{i}', f'L0_{n_idx}', color=color, penwidth=penwidth)
            
    # 4. Draw the Edges between Hidden/Output Layers
    for l_idx in range(len(mlp.layers) - 1):
        for prev_n_idx in range(len(mlp.layers[l_idx].neurons)):
            for n_idx, neuron in enumerate(mlp.layers[l_idx+1].neurons):
                w = neuron.w[prev_n_idx].data
                color = 'blue' if w > 0 else 'red'
                penwidth = str(max(0.5, abs(w) * 3))
                dot.edge(f'L{l_idx}_{prev_n_idx}', f'L{l_idx+1}_{n_idx}', color=color, penwidth=penwidth)

    return dot
