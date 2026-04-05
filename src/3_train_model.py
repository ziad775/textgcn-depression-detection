import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import TextGCNModel
from preprocessing import load_and_clean_data

# --- Custom Graph Metrics ---
def masked_loss(y_true, y_pred, mask):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask) 
    loss *= mask
    return tf.reduce_mean(loss)

def masked_accuracy(y_true, y_pred, mask):
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy_all = tf.cast(correct_predictions, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def main():
    print("=== STEP 3: Model Training & Evaluation ===")
    
    # 1. Load the Offline Data
    print("Loading pre-computed X and A matrices...")
    doc_features = np.load("../data/doc_embeddings.npy")
    A_matrix = sp.load_npz("../data/A_matrix.npz")
    
    num_docs = doc_features.shape[0]
    total_nodes = A_matrix.shape[0]
    num_words = total_nodes - num_docs
    
    # Rebuild the X Matrix (Adding empty Word Nodes)
    word_features = np.zeros((num_words, 768))
    X_matrix = np.vstack([doc_features, word_features])
    X_tf = tf.convert_to_tensor(X_matrix, dtype=tf.float32)
    
    # Format the A Matrix
    A_coo = A_matrix.tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    A_tf = tf.sparse.SparseTensor(
        indices=indices,
        values=A_coo.data.astype(np.float32),
        dense_shape=A_coo.shape
    )
    A_tf = tf.sparse.reorder(A_tf)
    
    # ---------------------------------------------------------
    # REAL DATA LOGIC: Labels and Scientific Splitting
    # ---------------------------------------------------------
    
    # 2. Extract Real Labels from the CSV
    csv_path = "../data/twitter_English.csv" 
    print(f"Extracting true labels from {csv_path}...")
    
    # We pass it through the cleaner to guarantee the rows match our graph 1-to-1
    df = load_and_clean_data(csv_path)
    raw_labels = df['label'].values 
    
    # One-hot encode the document labels and pad for word nodes
    doc_labels = tf.one_hot(raw_labels, depth=2).numpy()
    word_labels = np.zeros((num_words, 2))
    Y_matrix = np.vstack([doc_labels, word_labels])
    Y_tf = tf.convert_to_tensor(Y_matrix, dtype=tf.float32)
    
    # 3. Scientific Data Splitting (70% Train, 10% Val, 20% Test)
    print("Generating Scientific Train/Val/Test splits...")
    doc_indices = np.arange(num_docs)
    
    # First split: Separate out 20% for the Test Set
    train_val_idx, test_idx = train_test_split(doc_indices, test_size=0.20, random_state=42)
    
    # Second split: Out of the remaining 80%, take 1/8th (which equals 10% of total) for Validation
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.125, random_state=42)
    
    # Create the empty masks for the whole graph
    train_mask = np.zeros(total_nodes, dtype=bool)
    val_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    
    # Activate the specific document nodes in each mask
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    train_mask_tf = tf.convert_to_tensor(train_mask)
    val_mask_tf = tf.convert_to_tensor(val_mask)
    test_mask_tf = tf.convert_to_tensor(test_mask)
    
    print(f"Data Split -> Training Docs: {len(train_idx)} | Validation Docs: {len(val_idx)} | Test Docs: {len(test_idx)}")
    
    # ---------------------------------------------------------
    # RESUME STANDARD TRAINING LOGIC
    # ---------------------------------------------------------

    # 4. Initialize Model & Optimizer
    model = TextGCNModel(num_classes=2, hidden_dim=200, dropout_rate=0.5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    # Prepare directories for saving weights
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
    
    # 5. The Training Loop with Checkpointing
    print("\nCommencing Scientific Training (200 Epochs)...")
    epochs = 200
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            predictions = model([X_tf, A_tf], training=True)
            loss = masked_loss(Y_tf, predictions, train_mask_tf)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        # Calculate Training and Validation metrics
        train_acc = masked_accuracy(Y_tf, predictions, train_mask_tf)
        
        # Run validation pass without dropout (training=False)
        val_preds = model([X_tf, A_tf], training=False)
        val_loss = masked_loss(Y_tf, val_preds, val_mask_tf)
        val_acc = masked_accuracy(Y_tf, val_preds, val_mask_tf)
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights(checkpoint_path)
            
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 6. Final Evaluation (Scientific Metrics)
    print("\n--- OFFICIAL SCIENTIFIC EVALUATION ---")
    print("Loading the best model weights for the test set...")
    model.load_weights(checkpoint_path)
    
    # 1. Get the raw probability predictions for the whole graph
    test_preds_probs = model([X_tf, A_tf], training=False)
    
    # 2. Extract ONLY the nodes that belong to our blind Test Set
    test_mask_indices = np.where(test_mask)[0]
    
    # 3. Convert probabilities to final class guesses (0 or 1)
    y_true_test = np.argmax(Y_matrix[test_mask_indices], axis=1)
    y_pred_test = np.argmax(test_preds_probs.numpy()[test_mask_indices], axis=1)
    
    # 4. Generate the full scientific report
    print("\n=== CLASSIFICATION REPORT ===")
    target_names = ['Class 0 (Non-Depressed)', 'Class 1 (Depressed)']
    report = classification_report(y_true_test, y_pred_test, target_names=target_names, zero_division=0)
    print(report)
    
    print("=== CONFUSION MATRIX ===")
    cm = confusion_matrix(y_true_test, y_pred_test)
    print(cm)

if __name__ == "__main__":
    main()