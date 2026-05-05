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
# Notice we added 'training=False' here
    def call(self, inputs, training=False):
        x, a = inputs
        
        # 1. Split the data
        x_expert1 = x[:, :768]
        x_expert2 = x[:, 768:]

        # 2. Get Expert 1 Probabilities
        out_expert1 = self.gcn1_expert1([x_expert1, a])
        # Dropout behaves differently during training vs testing automatically!
        out_expert1 = self.dropout_expert1(out_expert1, training=training)
        probs_expert1 = self.classifier_expert1([out_expert1, a]) 

        # 3. Get Expert 2 Probabilities
        out_expert2 = self.gcn1_expert2([x_expert2, a])
        out_expert2 = self.dropout_expert2(out_expert2, training=training)
        probs_expert2 = self.classifier_expert2([out_expert2, a]) 

        # ==========================================
        # 4. THE DUAL-LOGIC VOTING MECHANISM
        # ==========================================
        if training:
            # PHASE 1: TRAINING (The Calculus Protector)
            # We use Soft-Voting (Average) so both experts receive gradient updates and learn equally.
            final_votes = (probs_expert1 + probs_expert2) / 2.0
            
        else:
            # PHASE 2: TESTING / INFERENCE (Your "OR Gate" Logic)
            # The model is no longer learning, so we can safely use the strict Max-Pooling logic!
            # If EITHER expert is highly confident, we trust them.
            final_votes = tf.maximum(probs_expert1, probs_expert2)
            
            # Re-normalize so the maximums still add up to 100%
            final_votes = final_votes / tf.reduce_sum(final_votes, axis=1, keepdims=True)
            
        return final_votes