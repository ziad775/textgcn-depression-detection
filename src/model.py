import tensorflow as tf
from spektral.layers import GCNConv

class VotingTextGCNModel(tf.keras.Model):
    def __init__(self, num_classes=2, hidden_dim=200, dropout_rate=0.5):
        super().__init__()
        print(f"--- Initializing DUAL-BRANCH Voting GCN Architecture ---")
        print(f"-> Expert 1 Branch (MentalBERT) hidden dim: {hidden_dim}")
        print(f"-> Expert 2 Branch (RoBERTaDepression) hidden dim: {hidden_dim}")
        print(f"-> Final Layer: Soft Voting Mechanism (Average)")
        
        # ==========================================
        # EXPERT 1 BRANCH (MentalBERT Focus)
        # ==========================================
        self.gcn1_expert1 = GCNConv(hidden_dim, activation='relu')
        self.dropout_expert1 = tf.keras.layers.Dropout(dropout_rate)
        self.classifier_expert1 = GCNConv(num_classes, activation='softmax')

        # ==========================================
        # EXPERT 2 BRANCH (RoBERTaDepression Focus)
        # ==========================================
        self.gcn1_expert2 = GCNConv(hidden_dim, activation='relu')
        self.dropout_expert2 = tf.keras.layers.Dropout(dropout_rate)
        self.classifier_expert2 = GCNConv(num_classes, activation='softmax')

    def call(self, inputs):
        """
        The forward pass of the neural network.
        Splits the concatenated features, runs parallel networks, and votes.
        """
        # x is the massive 1536-dimensional matrix, a is the adjacency matrix
        x, a = inputs
        
        # 1. THE SPLIT: Separate the 1536d vector back into two 768d vectors
        # (Assuming you concatenated them as [MentalBERT(768) + RoBERTaDepression(768)])
        x_expert1 = x[:, :768]   # Grabs the first 768 columns
        x_expert2 = x[:, 768:]   # Grabs the last 768 columns

        # 2. EXPERT 1 DIAGNOSIS
        out_expert1 = self.gcn1_expert1([x_expert1, a])
        out_expert1 = self.dropout_expert1(out_expert1)
        probs_expert1 = self.classifier_expert1([out_expert1, a]) # Outputs e.g., [0.20, 0.80]

        # 3. EXPERT 2 DIAGNOSIS
        out_expert2 = self.gcn1_expert2([x_expert2, a])
        out_expert2 = self.dropout_expert2(out_expert2)
        probs_expert2 = self.classifier_expert2([out_expert2, a]) # Outputs e.g., [0.40, 0.60]

        # 4. THE VOTING MECHANISM (Soft Voting / Averaging)
        # We add the probabilities together and divide by 2.
        # Example: ([0.20, 0.80] + [0.40, 0.60]) / 2 = [0.30, 0.70] -> Final Prediction: Class 1
        final_votes = (probs_expert1 + probs_expert2) / 2.0
        
        return final_votes