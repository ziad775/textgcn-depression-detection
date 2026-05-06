import os
import gc
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model import TextGCNModel

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
    print("=== STEP 3: Model Training (Homogeneous SMOTE Graph) ===")
    
    # 1. Load the NEW SMOTE-Balanced Data
    print("Loading balanced features and Homogeneous Adjacency Matrix...")
    doc_features = np.load("../data/balanced_doc_embeddings.npy")
    raw_labels = np.load("../data/balanced_labels.npy")
    A_matrix = sp.load_npz("../data/A_matrix.npz")
    
    num_nodes = doc_features.shape[0]
    print(f"Total Nodes in Graph: {num_nodes} (No Word Nodes)")
    
    # 2. Convert to TensorFlow Tensors
    X_tf = tf.convert_to_tensor(doc_features, dtype=tf.float32)
    
    A_coo = A_matrix.tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    A_tf = tf.sparse.SparseTensor(
        indices=indices,
        values=A_coo.data.astype(np.float32),
        dense_shape=A_coo.shape
    )
    A_tf = tf.sparse.reorder(A_tf)
    
    doc_labels = tf.one_hot(raw_labels, depth=2).numpy()
    Y_tf = tf.convert_to_tensor(doc_labels, dtype=tf.float32)
    
    # ==========================================
    # PHASE 4: 5-FOLD CROSS-VALIDATION SETUP
    # ==========================================
    print("\nExecuting Phase 4: Initializing 5-Fold Splits...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # --- NEW: Added fold_train_accs tracker ---
    fold_accs, fold_precs, fold_recs, fold_f1s, fold_train_accs = [], [], [], [], []
    
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

    with tf.device('/CPU:0'):
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_nodes))):
            print(f"\n==================================================")
            print(f"              STARTING FOLD {fold + 1} OF 5")
            print(f"==================================================")
            
            tf.keras.backend.clear_session()
            gc.collect()

            # Create masks for the specific fold
            train_mask = np.zeros(num_nodes, dtype=bool)
            test_mask = np.zeros(num_nodes, dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            train_mask_tf = tf.convert_to_tensor(train_mask)
            test_mask_tf = tf.convert_to_tensor(test_mask)
            
            # Using hidden_dim=128 for memory safety with the dense semantic graph
            model = TextGCNModel(num_classes=2, hidden_dim=128, dropout_rate=0.5, use_third_layer=False)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, decay=0.0) 

            epochs = 200
            best_test_acc = 0.0
            patience = 30
            patience_counter = 0
            
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    predictions = model([X_tf, A_tf], training=True)
                    loss = masked_loss(Y_tf, predictions, train_mask_tf)
                    
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                train_acc = masked_accuracy(Y_tf, predictions, train_mask_tf)
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
                    print(f"\n[EARLY STOPPING] Fold {fold+1} halted at Epoch {epoch}")
                    break
                
                gc.collect()
            
            # --- EVALUATE THIS FOLD ---
            model.load_weights(checkpoint_path)
            final_preds_probs = model([X_tf, A_tf], training=False)
            
            # --- NEW: Calculate final Training Metrics for this fold ---
            train_mask_indices = np.where(train_mask)[0]
            y_true_train = np.argmax(Y_tf.numpy()[train_mask_indices], axis=1)
            y_pred_train = np.argmax(final_preds_probs.numpy()[train_mask_indices], axis=1)
            final_train_acc = accuracy_score(y_true_train, y_pred_train)
            fold_train_accs.append(final_train_acc)
            
            # Calculate final Testing Metrics for this fold
            test_mask_indices = np.where(test_mask)[0]
            y_true_test = np.argmax(Y_tf.numpy()[test_mask_indices], axis=1)
            y_pred_test = np.argmax(final_preds_probs.numpy()[test_mask_indices], axis=1)
            
            acc = accuracy_score(y_true_test, y_pred_test)
            prec = precision_score(y_true_test, y_pred_test, average='macro', zero_division=0)
            rec = recall_score(y_true_test, y_pred_test, average='macro', zero_division=0)
            f1 = f1_score(y_true_test, y_pred_test, average='macro', zero_division=0)
            
            fold_accs.append(acc)
            fold_precs.append(prec)
            fold_recs.append(rec)
            fold_f1s.append(f1)
            
            print(f"-> Fold {fold+1} Completed | Train Acc: {final_train_acc:.4f} | Test Acc: {acc:.4f} | Test F1: {f1:.4f}")

    print("\n==================================================")
    print("      FINAL 5-FOLD CROSS-VALIDATION METRICS       ")
    print("==================================================")
    print(f"Train Accuracy: {np.mean(fold_train_accs):.4f} (± {np.std(fold_train_accs):.4f})")
    print(f"Test Accuracy:  {np.mean(fold_accs):.4f} (± {np.std(fold_accs):.4f})")
    print(f"Test Precision: {np.mean(fold_precs):.4f} (± {np.std(fold_precs):.4f})")
    print(f"Test Recall:    {np.mean(fold_recs):.4f} (± {np.std(fold_recs):.4f})")
    print(f"Test F1-Score:  {np.mean(fold_f1s):.4f} (± {np.std(fold_f1s):.4f})")
    print("==================================================")

if __name__ == "__main__":
    main()