import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from collections import defaultdict

class TextGCNGraph:
    def __init__(self, df: pd.DataFrame):
        print("\n--- Initializing Graph Builder ---")
        self.df = df
        self.num_docs = len(df)
        
        # Override default tokenizer to preserve emojis
        def custom_tokenizer(text):
            return text.split()
            
        # FIX 1: Turn off L2 normalization and use ALL vocabulary features 
        self.vectorizer = TfidfVectorizer(
            tokenizer=custom_tokenizer, 
            lowercase=False, 
            norm=None,          
            max_features=None,
            max_df=0.85    
        )
        
    def build_tfidf_edges(self):
        """
        Calculates TF-IDF to create edges between Word Nodes and Document Nodes.
        """
        print("Calculating TF-IDF (Word-Document edges)...")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_text'])
        
        self.vocab = self.vectorizer.get_feature_names_out()
        self.num_vocab = len(self.vocab)
        self.total_nodes = self.num_docs + self.num_vocab
        
        print(f"Graph Dimensions Locked:")
        print(f"-> Document Nodes: {self.num_docs}")
        print(f"-> Word Nodes:     {self.num_vocab}")
        print(f"-> Total Nodes:    {self.total_nodes}")
        
        return self.tfidf_matrix

    def build_pmi_edges(self, window_size=20):
        """
        Calculates Pointwise Mutual Information (PMI) to create Word-Word edges.
        """
        print(f"Calculating PMI (Word-Word edges) with window size {window_size}...")
        windows = []
        
        vocab_set = set(self.vocab)
        
        for text in self.df['cleaned_text']:
            words = [w for w in text.split() if w in vocab_set]
            
            length = len(words)
            if length <= window_size:
                windows.append(set(words))
            else:
                for i in range(length - window_size + 1):
                    windows.append(set(words[i: i + window_size]))

        word_window_freq = defaultdict(int)
        word_pair_window_freq = defaultdict(int)
        total_windows = len(windows)

        for window in windows:
            for word in window:
                word_window_freq[word] += 1
            
            window_list = list(window)
            for i in range(len(window_list)):
                for j in range(i + 1, len(window_list)):
                    w1, w2 = window_list[i], window_list[j]
                    if w1 > w2:
                        w1, w2 = w2, w1
                    word_pair_window_freq[(w1, w2)] += 1

        pmi_edges = {}
        for (w1, w2), freq in word_pair_window_freq.items():
            p_i = word_window_freq[w1] / total_windows
            p_j = word_window_freq[w2] / total_windows
            p_i_j = freq / total_windows
            
            pmi = math.log(p_i_j / (p_i * p_j))
            
            if pmi > 0:
                pmi_edges[(w1, w2)] = pmi

        print(f"-> Generated {len(pmi_edges)} positive Word-Word connections.")
        return pmi_edges

    def build_jaccard_edges(self, threshold=0.0):
        """
        Calculates Jaccard Similarity to create Document-Document edges.
        """
        print(f"Calculating Jaccard Similarity (Doc-Doc edges) with threshold {threshold}...")
        
        doc_sets = [set(text.split()) for text in self.df['cleaned_text']]
        jaccard_edges = {}
        
        for i in range(self.num_docs):
            for j in range(i + 1, self.num_docs):
                set_i = doc_sets[i]
                set_j = doc_sets[j]
                
                intersection = len(set_i.intersection(set_j))
                
                if intersection > 0:
                    union = len(set_i.union(set_j))
                    jaccard = intersection / union
                    
                    if jaccard >= threshold:
                        jaccard_edges[(i, j)] = jaccard
                        
        print(f"-> Generated {len(jaccard_edges)} positive Doc-Doc connections.")
        return jaccard_edges

    def get_node_id_maps(self):
        """Maps documents and words to specific integer indices."""
        doc_ids = {f"Doc_{i}": i for i in range(self.num_docs)}
        word_ids = {word: i + self.num_docs for i, word in enumerate(self.vocab)}
        return doc_ids, word_ids

    # FIX 3: Add Equation 1 Normalization
    def normalize_adjacency(self, adj):
        """Applies the D^(-1/2) * A * D^(-1/2) normalization from Equation 1"""
        print("Applying Symmetric Normalization (Equation 1)...")
        rowsum = np.array(adj.sum(1))
        
        # Avoid division by zero
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        # D^(-1/2) * A * D^(-1/2)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

    def build_adjacency_matrix(self, pmi_edges, jaccard_edges):
        """
        Fuses TF-IDF, PMI, Jaccard edges, and self-loops into the master Adjacency Matrix (A).
        """
        print("\nAssembling Master Adjacency Matrix [A]...")
        
        row = []
        col = []
        weight = []
        
        doc_ids, word_ids = self.get_node_id_maps()
        
        # 1. Inject TF-IDF (Doc <-> Word Edges)
        coo_tfidf = self.tfidf_matrix.tocoo()
        for d, w, val in zip(coo_tfidf.row, coo_tfidf.col, coo_tfidf.data):
            word_idx = w + self.num_docs 
            row.extend([d, word_idx])
            col.extend([word_idx, d])
            weight.extend([val, val])

        # 2. Inject PMI (Word <-> Word Edges)
        for (w1, w2), pmi_val in pmi_edges.items():
            if w1 in word_ids and w2 in word_ids:
                id1 = word_ids[w1]
                id2 = word_ids[w2]
                row.extend([id1, id2])
                col.extend([id2, id1])
                weight.extend([pmi_val, pmi_val])
            
        # 3. Inject Jaccard (Doc <-> Doc Edges)
        for (d1, d2), jaccard_val in jaccard_edges.items():
            row.extend([d1, d2])
            col.extend([d2, d1])
            weight.extend([jaccard_val, jaccard_val])
            
        # 4. Inject Self-Loops (Node <-> Node)
        for i in range(self.total_nodes):
            row.append(i)
            col.append(i)
            weight.append(1.0)
            
        # 5. Construct Sparse Matrix
        adj_matrix = sp.csr_matrix(
            (weight, (row, col)), 
            shape=(self.total_nodes, self.total_nodes)
        )
        
        # Apply Equation 1 before returning!
        normalized_adj = self.normalize_adjacency(adj_matrix)
        
        print(f"-> Master Adjacency Matrix Built and Normalized! Shape: {normalized_adj.shape}")
        print(f"-> Total non-zero edges recorded: {normalized_adj.nnz}")
        return normalized_adj