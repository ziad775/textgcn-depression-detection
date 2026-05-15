import tensorflow as tf
import numpy as np

class TextGCNExplainer(tf.Module):
    def __init__(self, trained_model, num_features, num_edges):
        super().__init__()
        print("\n--- Initializing GNNExplainer ---")
        self.trained_model = trained_model
        
        # 1. THE MASKS (These are what the Explainer actually trains)
        # We start them with random noise. 
        # A value approaching 1.0 means "Keep this edge/feature"
        # A value approaching 0.0 means "Delete this edge/feature"
        self.feat_mask = tf.Variable(tf.random.normal([num_features]), name="feat_mask")
        self.edge_mask = tf.Variable(tf.random.normal([num_edges]), name="edge_mask")

    def call(self, x, a_sparse):
        """
        Applies the masks to the raw data, then feeds it to your TextGCNModel.
        """
        # Apply Sigmoid to force all mask weights strictly between 0.0 and 1.0
        feat_mask_sig = tf.nn.sigmoid(self.feat_mask)
        edge_mask_sig = tf.nn.sigmoid(self.edge_mask)

        # MASK THE FEATURES: 
        # Multiplies the 2304-dim vectors by the feature mask
        masked_x = x * feat_mask_sig

        # MASK THE EDGES:
        # Multiplies the mathematical weights of the TF-IDF/PMI/Cosine edges by the edge mask
        masked_a_values = a_sparse.values * edge_mask_sig
        
        # Rebuild the Sparse Adjacency Matrix with the "deleted" edges
        masked_a = tf.sparse.SparseTensor(
            indices=a_sparse.indices, 
            values=masked_a_values, 
            dense_shape=a_sparse.dense_shape
        )

        # Feed the dynamically pruned graph into your exact architecture
        return self.trained_model([masked_x, masked_a])

def explain_patient_diagnosis(explainer, x, a_sparse, target_node_idx, target_class, epochs=200):
    """
    The Optimization Loop. Trains the Explainer to find the critical subgraph.
    """
    print(f"\n[XAI] Beginning Explanation Extraction for Document Node: {target_node_idx}")
    print(f"[XAI] Target Class (0=Non-Depressed, 1=Depressed): {target_class}")
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # 1. Forward pass through the Explainer (which calls your TextGCN)
            predictions = explainer.call(x, a_sparse)
            
            # 2. Look ONLY at the prediction for the specific patient we are explaining
            patient_pred = tf.expand_dims(predictions[target_node_idx], 0)
            true_label = tf.constant([target_class])
            
            # 3. CORE EXPLAINER MATH: 
            # Loss = Classification Error + (Size Penalty on Masks)
            # We add L1 regularization (tf.reduce_sum) to force the masks to delete 
            # as many useless edges and features as possible.
            pred_loss = loss_fn(true_label, patient_pred)
            size_loss = 0.005 * tf.reduce_sum(tf.nn.sigmoid(explainer.edge_mask)) + \
                        0.005 * tf.reduce_sum(tf.nn.sigmoid(explainer.feat_mask))
            
            total_loss = pred_loss + size_loss

        # Calculate gradients and update the MASKS (NOT your model weights)
        gradients = tape.gradient(total_loss, explainer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, explainer.trainable_variables))

        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Total Loss: {total_loss.numpy():.4f} | Pred Loss: {pred_loss.numpy():.4f}")

    print("\n[XAI] Explanation extraction complete!")
    
    # Extract the final finalized masks
    final_edge_weights = tf.nn.sigmoid(explainer.edge_mask).numpy()
    final_feat_weights = tf.nn.sigmoid(explainer.feat_mask).numpy()
    
    return final_edge_weights, final_feat_weights