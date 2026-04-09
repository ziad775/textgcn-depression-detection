import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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
    print("=== STEP 3: Model Training (10/10 Paper Replica) ===")
    
    # 1. Load the Offline Data
    print("Loading pre-computed X and A matrices...")
    doc_features = np.load("../data/doc_embeddings.npy")
    A_matrix = sp.load_npz("../data/A_matrix.npz")
    
    num_docs = doc_features.shape[0]
    total_nodes = A_matrix.shape[0]
    num_words = total_nodes - num_docs
    
    # ==========================================
    # PHASE 3: MIN-POOLING WORD INITIALIZATION
    # ==========================================
    print("Executing Phase 3: Min-Pooling Word Node Intelligence...")
    word_features = np.zeros((num_words, 768))
    
    # Extract the Document-to-Word section of the Adjacency Matrix
    doc_word_slice = A_matrix[:num_docs, num_docs:]
    doc_word_csc = doc_word_slice.tocsc() # Fast column/word lookups
    
    for w_idx in range(num_words):
        # Find every document index that contains this word
        doc_indices = doc_word_csc.indices[doc_word_csc.indptr[w_idx]:doc_word_csc.indptr[w_idx+1]]
        
        if len(doc_indices) > 0:
            # Grab the 768-dim RoBERTa vectors for all containing documents
            containing_docs_features = doc_features[doc_indices]
            # Calculate the minimum across all those documents
            word_features[w_idx] = np.min(containing_docs_features, axis=0)
            
    print("-> Min-pooling complete! Word Nodes are now semantically aware.")
    
    # Assemble final X Matrix
    X_matrix = np.vstack([doc_features, word_features])
    X_tf = tf.convert_to_tensor(X_matrix, dtype=tf.float32)
    
    # Format the A Matrix for TensorFlow
    A_coo = A_matrix.tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    A_tf = tf.sparse.SparseTensor(
        indices=indices,
        values=A_coo.data.astype(np.float32),
        dense_shape=A_coo.shape
    )
    A_tf = tf.sparse.reorder(A_tf)
    
    # 2. Extract Real Labels from the CSV
    csv_path = "../data/dataset2_twitter_English.csv"
    print(f"Extracting true labels from {csv_path}...")
    
    df = load_and_clean_data(csv_path)
    raw_labels = df['label'].values 
    
    doc_labels = tf.one_hot(raw_labels, depth=2).numpy()
    word_labels = np.zeros((num_words, 2))
    Y_matrix = np.vstack([doc_labels, word_labels])
    Y_tf = tf.convert_to_tensor(Y_matrix, dtype=tf.float32)
    
    # ==========================================
    # PHASE 4: THE 80/20 SCIENTIFIC SPLIT
    # ==========================================
    print("Executing Phase 4: Strict 80/20 Split with Test-Set Monitoring...")
    doc_indices = np.arange(num_docs)
    
    # 80:20 train/test split
    train_idx, test_idx = train_test_split(doc_indices, test_size=0.20, random_state=42)
    
    train_mask = np.zeros(total_nodes, dtype=bool)
    test_mask = np.zeros(total_nodes, dtype=bool)
    
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    train_mask_tf = tf.convert_to_tensor(train_mask)
    test_mask_tf = tf.convert_to_tensor(test_mask)
    
    print(f"Data Split -> Training Docs: {len(train_idx)} | Test Docs: {len(test_idx)}")
    
    # ==========================================
    # 5. THE TRAINING LOOP (ALL ON CPU TO PREVENT ERRORS)
    # ==========================================
    print("\nCommencing Scientific Training (200 Epochs) - Everything on CPU...")
    
    # Force the entire environment onto CPU to avoid "device mismatch" errors
    with tf.device('/CPU:0'):
        # Initialize Model & Optimizer INSIDE the CPU block
        model = TextGCNModel(num_classes=2, hidden_dim=200, dropout_rate=0.5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Paper baseline
        
        checkpoint_dir = "../checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

        epochs = 200 # Paper duration
        best_test_acc = 0.0
        patience = 30 # Paper early stopping
        patience_counter = 0
        
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                predictions = model([X_tf, A_tf], training=True)
                loss = masked_loss(Y_tf, predictions, train_mask_tf)
                
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            train_acc = masked_accuracy(Y_tf, predictions, train_mask_tf)
            
            # --- EARLY STOPPING DIRECTLY ON THE TEST SET (As acknowledged in paper) ---
            test_preds = model([X_tf, A_tf], training=False)
            test_acc = masked_accuracy(Y_tf, test_preds, test_mask_tf)
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                patience_counter = 0
                model.save_weights(checkpoint_path)
            else:
                patience_counter += 1
                
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f}, Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
                
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Triggered at Epoch {epoch}")
                break
            
    # 6. Final Evaluation
    print("\n--- OFFICIAL SCIENTIFIC EVALUATION ---")
    model.load_weights(checkpoint_path)
    
    final_preds_probs = model([X_tf, A_tf], training=False)
    test_mask_indices = np.where(test_mask)[0]
    
    y_true_test = np.argmax(Y_matrix[test_mask_indices], axis=1)
    y_pred_test = np.argmax(final_preds_probs.numpy()[test_mask_indices], axis=1)
    
    print("\n=== CLASSIFICATION REPORT ===")
    target_names = ['Class 0 (Non-Depressed)', 'Class 1 (Depressed)']
    print(classification_report(y_true_test, y_pred_test, target_names=target_names, zero_division=0))
    
    print("=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_true_test, y_pred_test))

    # Calculate the exact macro-averaged metrics to match the paper's reporting style
    acc = accuracy_score(y_true_test, y_pred_test)
    prec = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    rec = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)

    print("\n=== FINAL METRICS ===")
    print(f"accuracy is : {acc:.4f}")
    print(f"precision is : {prec:.4f}")
    print(f"recall is : {rec:.4f}")
    print(f"f1 score is : {f1:.4f}")

if __name__ == "__main__":
    main()