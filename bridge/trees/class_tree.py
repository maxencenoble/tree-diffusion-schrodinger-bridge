from .config_tree import Graph


class BridgeTree(Graph):
    def __init__(self,
                 n_vertices,
                 datasets,
                 root_idx,
                 weight):
        super().__init__(n_vertices)
        self.graph[0].vertex.data = datasets.pop(root_idx)
        self.graph[1].vertex.data = datasets.pop()
        self.add_edge(0, 1, weight)


class BarycenterTree(Graph):
    def __init__(self,
                 n_vertices,
                 datasets,
                 barycenter_weights,
                 root_idx=None):
        super().__init__(n_vertices)
        if root_idx is None:
            self.graph[0].vertex.data = None
            leaf = 1
            barycenter_idx = 0
        else:
            self.add_edge(0, 1, barycenter_weights.pop(root_idx))
            self.graph[0].vertex.data = datasets.pop(root_idx)
            self.graph[1].vertex.data = None
            leaf = 2
            barycenter_idx = 1
        while datasets:
            self.add_edge(barycenter_idx, leaf, barycenter_weights.pop())
            self.graph[leaf].vertex.data = datasets.pop()
            leaf += 1
