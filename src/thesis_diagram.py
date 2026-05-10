import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.patches as patches

def generate_convolution_diagram():
    fig, ax = plt.subplots(figsize=(16, 8), facecolor='#F5F7FA')
    ax.set_facecolor('#F5F7FA')

    # 1. Define Nodes and Layout
    # We will build one layout, and mirror it on the right side
    base_pos = {
        'Doc1': (1, 8), 'Doc2': (4, 8), 'Doc3': (1, 2), 'Doc4': (4, 2),
        'Pain': (2.5, 6.5), 'Ignore': (1.5, 5), 'Love': (3.5, 5), 
        'Anxiety': (1.5, 3.5), 'Happy': (2.5, 2.5)
    }

    words = ['Pain', 'Ignore', 'Love', 'Anxiety', 'Happy']
    docs = ['Doc1', 'Doc2', 'Doc3', 'Doc4']

    # 2. Define the Edges
    tfidf_edges = [('Doc1', 'Pain'), ('Doc1', 'Ignore'), ('Doc2', 'Pain'), 
                   ('Doc2', 'Love'), ('Doc3', 'Anxiety'), ('Doc3', 'Ignore'),
                   ('Doc4', 'Happy'), ('Doc4', 'Love')]
    pmi_edges = [('Pain', 'Ignore'), ('Ignore', 'Anxiety'), 
                 ('Love', 'Happy'), ('Anxiety', 'Happy')]
    cosine_edges = [('Doc1', 'Doc2'), ('Doc3', 'Doc4'), ('Doc1', 'Doc3')]

    # 3. Create Left (Output) and Right (Input) Graphs
    G_left = nx.Graph()
    G_right = nx.Graph()
    
    pos_left = {k: (v[0], v[1]) for k, v in base_pos.items()}
    pos_right = {k: (v[0] + 8, v[1]) for k, v in base_pos.items()} # Shift right by 8 units

    # Add nodes to graphs - POSITIONS SWITCHED
    for node in docs:
        G_left.add_node(node, type='doc', label=f"R({node})") # Left is now Output
        G_right.add_node(node, type='doc', label=node)        # Right is now Input
    for node in words:
        G_left.add_node(node, type='word', label=f"R({node})") # Left is now Output
        G_right.add_node(node, type='word', label=node)        # Right is now Input

    # Add edges to both graphs
    for edge_list, e_type in [(tfidf_edges, 'tfidf'), (pmi_edges, 'pmi'), (cosine_edges, 'cosine')]:
        G_left.add_edges_from(edge_list, edge_type=e_type)
        G_right.add_edges_from(edge_list, edge_type=e_type)

    # 4. Drawing Function for Graphs
    def draw_graph(G, pos):
        # Draw Edges
        nx.draw_networkx_edges(G, pos, edgelist=tfidf_edges, edge_color='#4A90E2', width=2, style='dotted')
        nx.draw_networkx_edges(G, pos, edgelist=pmi_edges, edge_color='#E74C3C', width=2, style='dashed')
        nx.draw_networkx_edges(G, pos, edgelist=cosine_edges, edge_color='#2ECC71', width=2.5, style='dotted')
        
        # Draw Nodes
        doc_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'doc']
        word_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'word']
        
        nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes, node_color='#D2B4DE', node_shape='s', node_size=2000, edgecolors='black')
        nx.draw_networkx_nodes(G, pos, nodelist=word_nodes, node_color='#F9E79F', node_shape='o', node_size=2000, edgecolors='black')
        
        # Draw Labels
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, font_weight='bold')

    # Render Left and Right Graphs
    draw_graph(G_left, pos_left)
    draw_graph(G_right, pos_right)

    # 5. Draw the Center "Hidden Layers" Box
    center_x, center_y = 6.5, 5
    box_width, box_height = 2, 1.5
    rect = patches.Rectangle((center_x - box_width/2, center_y - box_height/2), box_width, box_height, 
                             linewidth=2, edgecolor='black', facecolor='white', linestyle='dashed', zorder=3)
    ax.add_patch(rect)
    plt.text(center_x, center_y, "Hidden\nLayers", ha='center', va='center', fontsize=12, fontweight='bold', zorder=4)

    # 6. Draw Convolution Lines (Connecting graphs to the center box)
    # Drawing sweeping lines from nodes to the center box to represent Message Passing
    for node in base_pos.keys():
        left_coord = pos_left[node]
        right_coord = pos_right[node]
        
        # Left to Box
        ax.plot([left_coord[0]+0.3, center_x - box_width/2], [left_coord[1], center_y], 
                color='gray', alpha=0.3, linewidth=1.5, zorder=1)
        # Box to Right
        ax.plot([center_x + box_width/2, right_coord[0]-0.3], [center_y, right_coord[1]], 
                color='gray', alpha=0.3, linewidth=1.5, zorder=1)

    # 7. Add Bounding Boxes and Titles - TITLES SWITCHED
    bbox_kwargs = dict(boxstyle="round,pad=0.5", edgecolor="black", facecolor="none", linewidth=2, linestyle='dashdot')
    
    # Left Box (Now Output)
    ax.add_patch(patches.Rectangle((-0.5, 0.5), 6, 9, linewidth=2, edgecolor='black', facecolor='none', linestyle='dashed'))
    plt.text(2.5, 0.0, "Word-Document Representations", ha='center', fontsize=12, fontweight='bold')
    
    # Right Box (Now Input)
    ax.add_patch(patches.Rectangle((7.5, 0.5), 6, 9, linewidth=2, edgecolor='black', facecolor='none', linestyle='dashed'))
    plt.text(10.5, 0.0, "Word-Document Graph", ha='center', fontsize=12, fontweight='bold')

    # 8. Custom Legend
    legend_elements = [
        mpatches.Patch(facecolor='#D2B4DE', edgecolor='black', label='Document Node'),
        mpatches.Patch(facecolor='#F9E79F', edgecolor='black', label='Word Node'),
        Line2D([0], [0], color='#4A90E2', lw=2, ls='dotted', label='TF-IDF Edge'),
        Line2D([0], [0], color='#E74C3C', lw=2, ls='dashed', label='PMI Edge'),
        Line2D([0], [0], color='#2ECC71', lw=2.5, ls='dotted', label='Cosine Similarity Edge (Semantic)')
    ]
    plt.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.15),
               ncol=5, frameon=False, fontsize=11)

    # 9. Polish and Export
    plt.axis('off')
    plt.tight_layout()
    
    save_path = "TextGCN_Convolution_Diagram.png"
    plt.savefig(save_path, dpi=400, bbox_inches='tight')
    print(f"Success! High-resolution architecture diagram saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    generate_convolution_diagram()