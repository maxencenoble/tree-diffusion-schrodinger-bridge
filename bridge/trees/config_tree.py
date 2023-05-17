from __future__ import annotations


class Vertex():

    def __init__(self,
                 idx=None,
                 data=None,
                 cache_dl=None,
                 save_dl_sde=None,
                 save_dl_ode=None,
                 mean=None,
                 var=None,
                 nb_samples=None):
        self.idx = idx
        self.data = data
        self.cache_dl = cache_dl
        self.save_dl_sde = save_dl_sde
        self.save_dl_ode = save_dl_ode
        self.mean = mean
        self.var = var
        self.nb_samples = nb_samples
        self.first_forward = False


class Edge():

    def __init__(self, weight=None, ipf=None):
        self.weight = weight
        self.ipf = ipf
        self.direction = 'f'

    def change_direction(self):
        if self.direction == 'f':
            self.direction = 'b'
        else:
            self.direction = 'f'


class Node():

    def __init__(self, *args, **kwargs):
        self.vertex = Vertex(*args, **kwargs)
        self.edges = dict()


class Graph():
    def __init__(self, n_vertices: int) -> None:
        self.n_vertices = n_vertices
        self.graph = [Node(idx=idx) for idx in range(n_vertices)]

    def add_edge(self, source: int, destination: int, *args):
        """Create an edge between `source` and `destination`.
        The weight of the edge is 'weight'.

        Args:
            source (int): source index.
            destination (int): destination index.
            weight (float): weight of the edge
        """
        self.graph[source].edges[destination] = Edge(*args)

    def revert_edge(self, source: int, destination: int):
        edge = self.graph[source].edges.pop(destination)
        edge.change_direction()
        self.graph[destination].edges[source] = edge

    def get_leaves(self) -> set[int]:
        """Get the set of leaves of a graph."""
        out = set()
        for i in range(len(self.graph)):
            if not self.graph[i].edges:
                out.add(i)
        return out

    def get_root(self) -> int:
        """Get the root of a graph. (graph is assumed to be a tree)"""
        nodes = set(list(range(self.n_vertices)))
        nodes_children = set()
        for vertex in self.graph:
            nodes_children = nodes_children.union(set(vertex.edges.keys()))
        out = nodes.difference(nodes_children)
        return list(out)[0]

    def find_path(self, source: int, destination: int) -> list[int]:
        """Find a path between `source` and `destination`.
        Note that source is not necessarily the root of `graph`."""
        lst = []
        find_path_rec(self, lst, source, destination)
        return lst[::-1]

    def revert_path(self, path: list[int]):
        """Inplace reversal of the edges along a path."""
        for idx in range(len(path) - 1):
            self.revert_edge(path[idx], path[idx + 1])

    def change_root(self, new_root: int):
        """Inplace change of the root of a tree."""
        root = self.get_root()
        path = self.find_path(root, new_root)
        self.revert_path(path)

    def __str__(self):
        strout = ''
        for idx in range(self.n_vertices):
            data_label = self.graph[idx].vertex.data
            if data_label is not None:
                strout += f'Node {idx} has label {data_label}. \n'
            else:
                strout += f'Node {idx} has no label. \n'
            if self.graph[idx].edges:
                strout += f'Node {idx} has children '
                for destination in self.graph[idx].edges.keys():
                    strout += f'{destination}, '
                strout = strout[:-2] + '.\n'
        return strout


def find_path_rec(graph: Graph, lst: list[int], root: int, target: int) -> bool:
    if root == target:
        lst.append(target)
        return True
    for child in graph.graph[root].edges.keys():
        out = find_path_rec(graph, lst, child, target)
        if out:
            lst.append(root)
            return True
    return False


if __name__ == '__main__':
    print('Example of tree with weights on the edges')
    graph = Graph(6)
    graph.add_edge(0, 1, 3.5)
    graph.add_edge(0, 2, 1.0)
    graph.add_edge(1, 3, 1.2)
    graph.add_edge(3, 4, 0.3)
    graph.add_edge(3, 5, 0.5)
    print(graph)
    path = graph.find_path(0, 5)
    print(path)
    print(graph.graph[0].edges[1].direction)
    graph.change_root(5)
    print(graph)
    print(graph.get_leaves())
    graph.change_root(5)
    print(graph)
    print(graph.graph[1].edges[0].direction)
