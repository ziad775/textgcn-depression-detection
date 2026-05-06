import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

class HomogeneousGraphBuilder:
    def __init__(self, doc_embeddings, threshold=0.85):
        """
        Initializes a Homogeneous Graph (Documents Only) for SMOTE compatibility.
        """
        print("\n--- Initializing Homogeneous Graph Builder (SMOTE-Compatible) ---")
        self.doc_embeddings = doc_embeddings
        self.total_nodes = doc_embeddings.shape[0]
        self.threshold = threshold
        
        print(f"Graph Dimensions Locked: {self.total_nodes} Document Nodes (0 Word Nodes)")

    def build_semantic_edges(self):
        """Calculates Cosine Similarity between all vectors (real and synthetic)."""
        print(f"Calculating Cosine Similarity Edges with strict threshold {self.threshold}...")
        
        # Calculate similarity for all pairs at once
        cos_sim_matrix = cosine_similarity(self.doc_embeddings)
        
        # Prevent self-loops (handled later by the identity matrix)
        np.fill_diagonal(cos_sim_matrix, 0)
        
        # Find coordinates above the threshold
        row_indices, col_indices = np.where(cos_sim_matrix >= self.threshold)
        
        semantic_edges = {}
        for i, j in zip(row_indices, col_indices):
            if i < j:  # Only store one direction to match undirected graph logic
                semantic_edges[(i, j)] = cos_sim_matrix[i, j]
                
        print(f"-> Discovered {len(semantic_edges)} Semantic Bridges!")
        return semantic_edges

    def normalize_adjacency(self, adj):
        """Applies symmetric normalization so the math doesn't explode during training."""
        print("Applying Symmetric Normalization (Equation 1)...")
        rowsum = np.array(adj.sum(1))
        
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

    def build_adjacency_matrix(self):
        """Fuses the Semantic Edges and Self-Loops into the final Adjacency Matrix."""
        print("\nAssembling Homogeneous Adjacency Matrix [A]...")
        
        semantic_edges = self.build_semantic_edges()
        row, col, weight = [], [], []
        
        # 1. Inject Semantic Bridges
        for (d1, d2), sim_val in semantic_edges.items():
            row.extend([d1, d2])
            col.extend([d2, d1])
            weight.extend([sim_val, sim_val])
            
        # 2. Inject Self-Loops (Every node must connect to itself)
        for i in range(self.total_nodes):
            row.append(i)
            col.append(i)
            weight.append(1.0)
            
        # 3. Construct Sparse Matrix
        adj_matrix = sp.csr_matrix(
            (weight, (row, col)), 
            shape=(self.total_nodes, self.total_nodes)
        )
        
        normalized_adj = self.normalize_adjacency(adj_matrix)
        
        print(f"-> Master Adjacency Matrix Built! Shape: {normalized_adj.shape}")
        print(f"-> Total non-zero edges recorded: {normalized_adj.nnz}")
        return normalized_adj