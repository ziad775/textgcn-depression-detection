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
            
        self.vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, lowercase=False)
        
    def build_tfidf_edges(self):
        """
        Calculates TF-IDF to create edges between Word Nodes and Document Nodes.
        """
        print("Calculating TF-IDF (Word-Document edges)...")
        # Attach tfidf_matrix to "self" so other functions can read it
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
        
        for text in self.df['cleaned_text']:
            words = text.split()
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

    def get_node_id_maps(self):
        """Maps documents and words to specific integer indices."""
        doc_ids = {f"Doc_{i}": i for i in range(self.num_docs)}
        word_ids = {word: i + self.num_docs for i, word in enumerate(self.vocab)}
        return doc_ids, word_ids

    def build_adjacency_matrix(self, pmi_edges):
        """
        Fuses TF-IDF edges, PMI edges, and self-loops into the master Adjacency Matrix (A).
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
            id1 = word_ids[w1]
            id2 = word_ids[w2]
            row.extend([id1, id2])
            col.extend([id2, id1])
            weight.extend([pmi_val, pmi_val])
            
        # 3. Inject Self-Loops (Node <-> Node)
        for i in range(self.total_nodes):
            row.append(i)
            col.append(i)
            weight.append(1.0)
            
        # 4. Construct Sparse Matrix
        adj_matrix = sp.csr_matrix(
            (weight, (row, col)), 
            shape=(self.total_nodes, self.total_nodes)
        )
        
        print(f"-> Master Adjacency Matrix Built! Shape: {adj_matrix.shape}")
        print(f"-> Total non-zero edges recorded: {adj_matrix.nnz}")
        return adj_matrix