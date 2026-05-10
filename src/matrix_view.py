import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_edge_types(matrix_path, num_docs, target_doc_indices=[0, 1, 2]):
    print("Loading Master Adjacency Matrix...")
    A_matrix = sp.load_npz(matrix_path)

    # 1. The Smart Probe: Gather a connected neighborhood of Nodes
    selected_nodes = set(target_doc_indices)
    
    for doc_id in target_doc_indices:
        row = A_matrix.getrow(doc_id)
        
        # Grab a few Semantic connections (Doc-Doc: index < num_docs)
        connected_docs = [col for col in row.indices if col < num_docs and col != doc_id]
        selected_nodes.update(connected_docs[:2]) 
        
        # Grab a few TF-IDF connections (Doc-Word: index >= num_docs)
        connected_words = [col for col in row.indices if col >= num_docs]
        selected_nodes.update(connected_words[:4]) 
        
    # Grab a few PMI connections (Word-Word) for the words we just extracted
    extracted_words = [n for n in selected_nodes if n >= num_docs]
    for w_id in extracted_words:
        row = A_matrix.getrow(w_id)
        connected_words = [col for col in row.indices if col >= num_docs and col != w_id]
        selected_nodes.update(connected_words[:2])
        
    # THE FIX: We MUST sort the list so Scipy matrix slicing and NetworkX ID mapping perfectly align
    selected_nodes = sorted(list(selected_nodes))
    
    # 2. Extract the Subgraph
    print(f"Extracting subgraph for {len(selected_nodes)} interconnected nodes...")
    sub_matrix = A_matrix[selected_nodes, :][:, selected_nodes]
    G = nx.from_scipy_sparse_array(sub_matrix)
    
    # Map the internal subgraph IDs back to the real Node IDs
    mapping = {i: selected_nodes[i] for i in range(len(selected_nodes))}
    G = nx.relabel_nodes(G, mapping)
    G.remove_edges_from(nx.selfloop_edges(G)) # Remove self-loops for a cleaner drawing
    
    # 3. Categorize Nodes and Color-Code Edges
    doc_nodes = [n for n in G.nodes() if n < num_docs]
    word_nodes = [n for n in G.nodes() if n >= num_docs]
    
    print(f"-> Found {len(doc_nodes)} Document Nodes.")
    print(f"-> Found {len(word_nodes)} Word Nodes.")
    
    edge_colors = []
    edge_labels = {}
    
    for u, v, d in G.edges(data=True):
        weight = d['weight']
        edge_labels[(u, v)] = f"{weight:.2f}"
        
        # The Mathematical Routing Logic
        if u < num_docs and v < num_docs:
            edge_colors.append('blue')    # Cosine Similarity (Doc-Doc)
        elif u >= num_docs and v >= num_docs:
            edge_colors.append('orange')  # PMI (Word-Word)
        else:
            edge_colors.append('green')   # TF-IDF (Doc-Word)

    # 4. Draw the Graph
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.9, seed=42) # k adjusts the spacing between nodes
    
    # Draw Document Nodes (Squares) and Word Nodes (Circles)
    if doc_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color='lightblue', 
                               node_shape='s', node_size=1200)
    if word_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='lightgreen', 
                               node_shape='o', node_size=800)
    
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    
    # Draw Edges and their Exact Weights
    if len(G.edges()) > 0:
        weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, alpha=0.7)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, font_color='black')
    
    # 5. Build the Legend for the Committee
    blue_patch = mpatches.Patch(color='blue', label='Cosine Similarity (Semantic Doc-Doc)')
    green_patch = mpatches.Patch(color='green', label='TF-IDF (Membership Doc-Word)')
    orange_patch = mpatches.Patch(color='orange', label='PMI (Contextual Word-Word)')
    doc_patch = plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=15, label='Document Node')
    word_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='Word Node')
    
    plt.legend(handles=[doc_patch, word_patch, blue_patch, green_patch, orange_patch], loc='upper left', fontsize=11)
    plt.title("TextGCN Architecture: The Three Mathematical Bridges", fontsize=16, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # IMPORTANT: Double-check that this number exactly matches the number of tweets
    # in the dataset you used to build A_matrix.npz!
    TOTAL_TWEETS = 1000 
    
    visualize_edge_types("../data/A_matrix.npz", num_docs=TOTAL_TWEETS)
