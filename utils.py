from abc import abstractmethod
import torch

class BaseLabelPropagation:
    """Base class for label propagation models.
    
    Parameters
    ----------
    adj_matrix: torch.FloatTensor
        Adjacency matrix of the graph.
    """
    def __init__(self, adj_matrix):
        self.norm_adj_matrix = self._normalize(adj_matrix)
        self.n_nodes = adj_matrix.size(0)
        self.one_hot_labels = None 
        self.n_classes = None
        self.labeled_mask = None
        self.predictions = None

    @staticmethod
    @abstractmethod
    def _normalize(adj_matrix):
        raise NotImplementedError("_normalize must be implemented")

    @abstractmethod
    def _propagate(self):
        raise NotImplementedError("_propagate must be implemented")

    def _one_hot_encode(self, labels):
        # Get the number of classes
        classes = torch.unique(labels)
        classes = classes[classes != -1]
        self.n_classes = classes.size(0)

        # One-hot encode labeled data instances and zero rows corresponding to unlabeled instances
        unlabeled_mask = (labels == -1)
        labels = labels.clone()  # defensive copying
        labels[unlabeled_mask] = 0
        self.one_hot_labels = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)
        self.one_hot_labels = self.one_hot_labels.scatter(1, labels.unsqueeze(1), 1)
        self.one_hot_labels[unlabeled_mask, 0] = 0

        self.labeled_mask = ~unlabeled_mask

    def fit(self, labels, max_iter, tol):
        """Fits a semi-supervised learning label propagation model.
        
        labels: torch.LongTensor
            Tensor of size n_nodes indicating the class number of each node.
            Unlabeled nodes are denoted with -1.
        max_iter: int
            Maximum number of iterations allowed.
        tol: float
            Convergence tolerance: threshold to consider the system at steady state.
        """
        self._one_hot_encode(labels)

        self.predictions = self.one_hot_labels.clone()
        prev_predictions = torch.zeros((self.n_nodes, self.n_classes), dtype=torch.float)

        for i in range(max_iter):
            # Stop iterations if the system is considered at a steady state
            variation = torch.abs(self.predictions - prev_predictions).sum().item()
            
            if variation < tol:
                print(f"The method stopped after {i} iterations, variation={variation:.4f}.")
                break

            prev_predictions = self.predictions
            self._propagate()

    def predict(self):
        return self.predictions

    def predict_classes(self):
        return self.predictions.max(dim=1).indices
    
class LabelPropagation(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

    @staticmethod
    def _normalize(adj_matrix):
        """Computes D^-1 * W"""
        degs = adj_matrix.sum(dim=1)
        degs[degs == 0] = 1  # avoid division by 0 error
        return adj_matrix / degs[:, None]

    def _propagate(self):
        self.predictions = torch.matmul(self.norm_adj_matrix, self.predictions)

        # Put back already known labels
        self.predictions[self.labeled_mask] = self.one_hot_labels[self.labeled_mask]

    def fit(self, labels, max_iter=1000, tol=1e-3):
        super().fit(labels, max_iter, tol)
        
class LabelSpreading(BaseLabelPropagation):
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)
        self.alpha = None

    @staticmethod
    def _normalize(adj_matrix):
        """Computes D^-1/2 * W * D^-1/2"""
        degs = adj_matrix.sum(dim=1)
        norm = torch.pow(degs, -0.5)
        norm[torch.isinf(norm)] = 1
        return adj_matrix * norm[:, None] * norm[None, :]

    def _propagate(self):
        self.predictions = (
            self.alpha * torch.matmul(self.norm_adj_matrix, self.predictions)
            + (1 - self.alpha) * self.one_hot_labels
        )
    
    def fit(self, labels, max_iter=1000, tol=1e-3, alpha=0.5):
        """
        Parameters
        ----------
        alpha: float
            Clamping factor.
        """
        self.alpha = alpha
        super().fit(labels, max_iter, tol)
        
        
def example():
    import pandas as pd
    import numpy as np
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create caveman graph
    n_cliques = 4
    size_cliques = 10
    caveman_graph = nx.connected_caveman_graph(n_cliques, size_cliques)
    adj_matrix = nx.adjacency_matrix(caveman_graph).toarray()

    # Create labels
    labels = np.full(n_cliques * size_cliques, -1.)

    # Only one node per clique is labeled. Each clique belongs to a different class.
    labels[0] = 0
    labels[size_cliques] = 1
    labels[size_cliques * 2] = 2
    labels[size_cliques * 3] = 3

    # Create input tensors
    adj_matrix_t = torch.FloatTensor(adj_matrix)
    labels_t = torch.LongTensor(labels)

    # Learn with Label Propagation
    label_propagation = LabelPropagation(adj_matrix_t)
    print("Label Propagation: ", end="")
    label_propagation.fit(labels_t)
    label_propagation_output_labels = label_propagation.predict_classes()

    # Learn with Label Spreading
    label_spreading = LabelSpreading(adj_matrix_t)
    print("Label Spreading: ", end="")
    label_spreading.fit(labels_t, alpha=0.8)
    label_spreading_output_labels = label_spreading.predict_classes()
    
    # Plot graphs
    color_map = {-1: "grey", 0: "blue", 1: "green", 2: "red", 3: "cyan"}
    input_labels_colors = [color_map[l] for l in labels]
    lprop_labels_colors = [color_map[l] for l in label_propagation_output_labels.numpy()]
    lspread_labels_colors = [color_map[l] for l in label_spreading_output_labels.numpy()]

    plt.figure(figsize=(14, 6))
    ax1 = plt.subplot(1, 4, 1)
    ax2 = plt.subplot(1, 4, 2)
    ax3 = plt.subplot(1, 4, 3)

    ax1.title.set_text("Raw data (4 classes)")
    ax2.title.set_text("Label Propagation")
    ax3.title.set_text("Label Spreading")

    pos = nx.spring_layout(caveman_graph)
    nx.draw(caveman_graph, ax=ax1, pos=pos, node_color=input_labels_colors, node_size=50)
    nx.draw(caveman_graph, ax=ax2, pos=pos, node_color=lprop_labels_colors, node_size=50)
    nx.draw(caveman_graph, ax=ax3, pos=pos, node_color=lspread_labels_colors, node_size=50)

    # Legend
    ax4 = plt.subplot(1, 4, 4)
    ax4.axis("off")
    legend_colors = ["grey", "blue", "green", "red", "cyan"]
    legend_labels = ["unlabeled", "class 0", "class 1", "class 2", "class 3"]
    dummy_legend = [ax4.plot([], [], ls='-', c=c)[0] for c in legend_colors]
    plt.legend(dummy_legend, legend_labels)

    plt.show()