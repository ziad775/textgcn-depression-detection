import os
import gc
import optuna # <--- THE MAGIC OPTIMIZER
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, accuracy_score
from model import TextGCNModel
from preprocessing import load_and_clean_data

# --- Custom Graph Metrics (Unchanged) ---
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

# --- Global Variables for Data ---
# We load the data OUTSIDE the Optuna loop so we don't reload 8GB of data 50 times!
doc_features = np.load("../data/doc_embeddings.npy")
A_matrix = sp.load_npz("../data/A_matrix.npz")
df = load_and_clean_data("../data/dataset2_twitter_English.csv")

num_docs = doc_features.shape[0]
total_nodes = A_matrix.shape[0]
num_words = total_nodes - num_docs

# Setup X Matrix (Min-Pooling)
word_features = np.zeros((num_words, 768))
doc_word_slice = A_matrix[:num_docs, num_docs:].tocsc() 
for w_idx in range(num_words):
    doc_indices = doc_word_slice.indices[doc_word_slice.indptr[w_idx]:doc_word_slice.indptr[w_idx+1]]
    if len(doc_indices) > 0:
        word_features[w_idx] = np.min(doc_features[doc_indices], axis=0)

X_tf = tf.convert_to_tensor(np.vstack([doc_features, word_features]), dtype=tf.float32)

# Setup A Matrix
A_coo = A_matrix.tocoo()
A_tf = tf.sparse.reorder(tf.sparse.SparseTensor(
    indices=np.column_stack((A_coo.row, A_coo.col)),
    values=A_coo.data.astype(np.float32),
    dense_shape=A_coo.shape
))

# Setup Labels
doc_labels = tf.one_hot(df['label'].values, depth=2).numpy()
Y_matrix = np.vstack([doc_labels, np.zeros((num_words, 2))])
Y_tf = tf.convert_to_tensor(Y_matrix, dtype=tf.float32)

# ==========================================
# OPTUNA OBJECTIVE FUNCTION
# ==========================================
def objective(trial):
    """
    This function represents ONE attempt by Optuna to find the best settings.
    """
    # 1. The Exact Search Space from the Research Paper
    num_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5])
    hidden_dim = trial.suggest_categorical("hidden_dim", [100, 200, 300, 400, 500])
    dropout_rate = trial.suggest_categorical("dropout_rate", [0.01, 0.05, 0.1, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [0.01, 0.02, 0.03, 0.04, 0.05])
    weight_decay = trial.suggest_categorical("weight_decay", [0, 0.005, 0.05])
    
    print(f"\n--- OPTUNA TRIAL {trial.number} ---")
    print(f"Testing: L={num_layers}, H={hidden_dim}, DR={dropout_rate}, LR={learning_rate}, WD={weight_decay}")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_f1s = []
    
    with tf.device('/CPU:0'):
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_docs))):
            
            tf.keras.backend.clear_session()
            gc.collect()

            train_mask = np.zeros(total_nodes, dtype=bool)
            test_mask = np.zeros(total_nodes, dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            train_mask_tf = tf.convert_to_tensor(train_mask)
            test_mask_tf = tf.convert_to_tensor(test_mask)
            
            # Feed Optuna's choices into the Model
            model = TextGCNModel(num_classes=2, num_layers=num_layers, hidden_dim=hidden_dim, dropout_rate=dropout_rate)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=weight_decay) 

            best_test_acc = 0.0
            patience_counter = 0
            
            for epoch in range(200):
                with tf.GradientTape() as tape:
                    predictions = model([X_tf, A_tf], training=True)
                    loss = masked_loss(Y_tf, predictions, train_mask_tf)
                    
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                test_preds = model([X_tf, A_tf], training=False)
                test_acc = masked_accuracy(Y_tf, test_preds, test_mask_tf)
                
                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= 10: # Paper's early stopping criterion
                    break
                
                gc.collect()
            
            # Evaluate this fold
            final_preds_probs = model([X_tf, A_tf], training=False)
            test_mask_indices = np.where(test_mask)[0]
            y_true_test = np.argmax(Y_matrix[test_mask_indices], axis=1)
            y_pred_test = np.argmax(final_preds_probs.numpy()[test_mask_indices], axis=1)
            
            fold_f1s.append(f1_score(y_true_test, y_pred_test, average='macro', zero_division=0))

    # Calculate the average F1 score across all 5 folds
    avg_f1 = np.mean(fold_f1s)
    print(f"Trial {trial.number} Result -> Average F1: {avg_f1:.4f}")
    
    # We want Optuna to MAXIMIZE this number
    return avg_f1

if __name__ == "__main__":
    print("=== STARTING OPTUNA HYPERPARAMETER SEARCH ===")
    
    # Create the Optuna Study
    study = optuna.create_study(direction="maximize")
    
    # RUN THE SEARCH! 
    # n_trials=10 means it will try 10 different combinations before stopping. 
    # You can increase this to 50 if you want to leave your Mac running overnight!
    study.optimize(objective, n_trials=10)
    
    print("\n==================================================")
    print("               OPTUNA SEARCH FINISHED             ")
    print("==================================================")
    print(f"Best F1-Score Achieved: {study.best_value:.4f}")
    print("Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")