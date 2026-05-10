import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        
    def build_tfidf_edges(self, top_k_words=30):
        """
        COMPRESSED FEATURE: Calculates TF-IDF but applies Top-K Sparsification.
        """
        print(f"Calculating Compressed TF-IDF (Max {top_k_words} Word edges per Document)...")
        
        raw_tfidf_matrix = self.vectorizer.fit_transform(self.df['cleaned_text'])
        
        new_data = []
        new_indices = []
        new_indptr = [0]
        
        for i in range(raw_tfidf_matrix.shape[0]):
            start = raw_tfidf_matrix.indptr[i]
            end = raw_tfidf_matrix.indptr[i+1]
            data_slice = raw_tfidf_matrix.data[start:end]
            idx_slice = raw_tfidf_matrix.indices[start:end]
            
            if len(data_slice) > top_k_words:
                best_k_indices = np.argsort(data_slice)[-top_k_words:]
                new_data.extend(data_slice[best_k_indices])
                new_indices.extend(idx_slice[best_k_indices])
            else:
                new_data.extend(data_slice)
                new_indices.extend(idx_slice)
                
            new_indptr.append(len(new_data))
            
        self.tfidf_matrix = sp.csr_matrix(
            (new_data, new_indices, new_indptr), 
            shape=raw_tfidf_matrix.shape
        )
        
        self.vocab = self.vectorizer.get_feature_names_out()
        self.num_vocab = len(self.vocab)
        self.total_nodes = self.num_docs + self.num_vocab
        
        print(f"Graph Dimensions Locked:")
        print(f"-> Document Nodes: {self.num_docs}")
        print(f"-> Word Nodes:     {self.num_vocab}")
        print(f"-> Total Nodes:    {self.total_nodes}")
        print(f"-> Edges successfully pruned! Retained the highest signal connections.")
        
        # --- DEBUG PRINT BLOCK FOR TF-IDF ---
        if new_data:
            print(f"   [DEBUG] TF-IDF Weights | Min: {np.min(new_data):.4f} | Max: {np.max(new_data):.4f} | Mean: {np.mean(new_data):.4f}")
            print(f"   [DEBUG] TF-IDF Sample: {new_data[:5]}")
        # ------------------------------------
        
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
            
            if pmi > 0.5:
                pmi_edges[(w1, w2)] = pmi

        print(f"-> Generated {len(pmi_edges)} positive Word-Word connections.")
        
        # --- DEBUG PRINT BLOCK FOR PMI ---
        if pmi_edges:
            pmi_vals = list(pmi_edges.values())
            print(f"   [DEBUG] PMI Weights    | Min: {np.min(pmi_vals):.4f} | Max: {np.max(pmi_vals):.4f} | Mean: {np.mean(pmi_vals):.4f}")
            print(f"   [DEBUG] PMI Sample: {pmi_vals[:5]}")
        # ---------------------------------
        
        return pmi_edges

    def build_semantic_doc_edges(self, doc_embeddings, top_k=5):
        """
        COMPRESSED FEATURE: Calculates Cosine Similarity, but uses K-Nearest Neighbors (KNN).
        """
        print(f"Calculating Compressed Semantic Edges (Top-{top_k} Nearest Neighbors)...")
        
        cos_sim_matrix = cosine_similarity(doc_embeddings)
        np.fill_diagonal(cos_sim_matrix, 0) 
        
        semantic_edges = {}
        
        for i in range(self.num_docs):
            sim_scores = cos_sim_matrix[i]
            top_k_indices = np.argsort(sim_scores)[-top_k:]
            
            for j in top_k_indices:
                score = sim_scores[j]
                if score > 0.50: 
                    idx_1, idx_2 = min(i, j), max(i, j)
                    semantic_edges[(idx_1, idx_2)] = score
                
        print(f"-> Discovered {len(semantic_edges)} highly compressed Semantic Bridges!")
        
        # --- DEBUG PRINT BLOCK FOR SEMANTIC EDGES ---
        if semantic_edges:
            sem_vals = list(semantic_edges.values())
            print(f"   [DEBUG] Semantic Weights| Min: {np.min(sem_vals):.4f} | Max: {np.max(sem_vals):.4f} | Mean: {np.mean(sem_vals):.4f}")
            print(f"   [DEBUG] Semantic Sample: {sem_vals[:5]}")
        # --------------------------------------------
        
        return semantic_edges    

    def get_node_id_maps(self):
        """Maps documents and words to specific integer indices."""
        doc_ids = {f"Doc_{i}": i for i in range(self.num_docs)}
        word_ids = {word: i + self.num_docs for i, word in enumerate(self.vocab)}
        return doc_ids, word_ids

    def normalize_adjacency(self, adj):
        """Applies the D^(-1/2) * A * D^(-1/2) normalization from Equation 1"""
        print("Applying Symmetric Normalization (Equation 1)...")
        rowsum = np.array(adj.sum(1))
        
        with np.errstate(divide='ignore'):
            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

    def build_adjacency_matrix(self, pmi_edges, semantic_edges=None):
        """
        Fuses TF-IDF, PMI, Semantic edges, and self-loops into the master Adjacency Matrix (A).
        """
        print("\nAssembling Master Adjacency Matrix [A]...")
        
        row, col, weight = [], [], []
        doc_ids, word_ids = self.get_node_id_maps()
        
        # 1. Inject TF-IDF
        coo_tfidf = self.tfidf_matrix.tocoo()
        for d, w, val in zip(coo_tfidf.row, coo_tfidf.col, coo_tfidf.data):
            word_idx = w + self.num_docs 
            row.extend([d, word_idx])
            col.extend([word_idx, d])
            weight.extend([val, val])

        # 2. Inject PMI
        for (w1, w2), pmi_val in pmi_edges.items():
            if w1 in word_ids and w2 in word_ids:
                id1 = word_ids[w1]
                id2 = word_ids[w2]
                row.extend([id1, id2])
                col.extend([id2, id1])
                weight.extend([pmi_val, pmi_val])
            
        # 3. Inject Semantic Bridges
        if semantic_edges:
            for (d1, d2), sim_val in semantic_edges.items():
                row.extend([d1, d2])
                col.extend([d2, d1])
                weight.extend([sim_val, sim_val])
            
        # 4. Inject Self-Loops
        for i in range(self.total_nodes):
            row.append(i)
            col.append(i)
            weight.append(1.0)
            
        # 5. Construct Sparse Matrix
        adj_matrix = sp.csr_matrix(
            (weight, (row, col)), 
            shape=(self.total_nodes, self.total_nodes)
        )
        
        # --- DEBUG PRINT BLOCK FOR RAW MATRIX ---
        print(f"   [DEBUG] RAW Master Matrix Weights | Min: {adj_matrix.data.min():.4f} | Max: {adj_matrix.data.max():.4f}")
        # ----------------------------------------
        
        normalized_adj = self.normalize_adjacency(adj_matrix)
        
        # --- DEBUG PRINT BLOCK FOR NORMALIZED MATRIX ---
        print(f"   [DEBUG] NORMALIZED Matrix Weights | Min: {normalized_adj.data.min():.6f} | Max: {normalized_adj.data.max():.6f}")
        # -----------------------------------------------
        
        print(f"-> Master Adjacency Matrix Built and Normalized! Shape: {normalized_adj.shape}")
        print(f"-> Total non-zero edges recorded: {normalized_adj.nnz}")
        return normalized_adj
