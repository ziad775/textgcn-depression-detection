import tensorflow as tf
from spektral.layers import GCNConv

class TextGCNModel(tf.keras.Model):
    # Notice the new 'use_third_layer=False' parameter here
    def __init__(self, num_classes=2, hidden_dim=200, dropout_rate=0.5, use_third_layer=False):
        super().__init__()
        print(f"--- Initializing TextGCN Architecture ---")
        print(f"-> Hidden Dimension: {hidden_dim}")
        print(f"-> Output Classes: {num_classes}")
        print(f"-> Architecture Depth: {'3 Layers (Experimental)' if use_third_layer else 'Standard 2 Layers'}")
        
        self.use_third_layer = use_third_layer
        
        # Layer 1: The First Hidden Feature Extractor (1-hop)
        self.gcn1 = GCNConv(hidden_dim, activation='relu')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        
        # OPTIONAL Layer 2: The Experimental Middle Layer (2-hop)
        # We only create this if you turn the flag to True
        if self.use_third_layer:
            self.gcn_extra = GCNConv(hidden_dim, activation='relu')
            self.dropout_extra = tf.keras.layers.Dropout(dropout_rate)
        
        # Final Layer: The Classifier (Depressed vs. Non-Depressed)
        # (If 2 layers, this is Layer 2. If 3 layers, this is Layer 3)
        self.gcn_final = GCNConv(num_classes, activation='softmax')

    def call(self, inputs):
        """
        The forward pass of the neural network. 
        Spektral expects a list containing [Node Features (X), Adjacency Matrix (A)]
        """
        x, a = inputs
        
        # --- 1st Ripple (Gather from direct neighbors) ---
        x = self.gcn1([x, a])
        x = self.dropout1(x)
        
        # --- 2nd Ripple (Gather from neighbors' neighbors) ---
        # This only executes if you turned the flag on!
        if self.use_third_layer:
            x = self.gcn_extra([x, a])
            x = self.dropout_extra(x)
        
        # --- Final Ripple (The Softmax Judge) ---
        x = self.gcn_final([x, a])
        
        return x