import os
import gc
# Force legacy Keras to ensure Adam optimizer accepts the 'decay' parameter smoothly
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from model import TextGCNModel
from preprocessing import load_and_clean_data

def masked_loss(y_true, y_pred, mask, class_weights):
    weights_per_node = tf.reduce_sum(y_true * class_weights, axis=1)
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = loss * weights_per_node
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
    print("=== STEP 3: Model Training (Tri-Brain Graph) ===")
    
    # ==========================================
    # THE MASTER TOGGLE SWITCH
    # ==========================================
    # Set to True to apply the Minority Penalty (Better Recall).
    # Set to False to run standard cross-entropy (Better Precision).
    ENABLE_CLASS_WEIGHTS = False
    
    # Optional: If ENABLE_CLASS_WEIGHTS is True, you can manually override the minority weight here. 
    # Set to None to let the algorithm calculate it automatically.
    MANUAL_MINORITY_WEIGHT = None
    # ==========================================
    
    print("Loading pre-computed X and A matrices...")
    doc_features = np.load("../data/doc_embeddings.npy")
    A_matrix = sp.load_npz("../data/A_matrix.npz")
    
    num_docs = doc_features.shape[0]
    total_nodes = A_matrix.shape[0]
    num_words = total_nodes - num_docs 
    
    print("Executing Phase 3: Min-Pooling Word Node Intelligence...")
    feature_dim = doc_features.shape[1] 
    word_features = np.zeros((num_words, feature_dim))
    
    doc_word_slice = A_matrix[:num_docs, num_docs:]
    doc_word_csc = doc_word_slice.tocsc() 
    
    for w_idx in range(num_words):
        doc_indices = doc_word_csc.indices[doc_word_csc.indptr[w_idx]:doc_word_csc.indptr[w_idx+1]]
        if len(doc_indices) > 0:
            containing_docs_features = doc_features[doc_indices]
            word_features[w_idx] = np.min(containing_docs_features, axis=0)
            
    X_matrix = np.vstack([doc_features, word_features])
    X_tf = tf.convert_to_tensor(X_matrix, dtype=tf.float32)
    
    A_coo = A_matrix.tocoo()
    indices = np.column_stack((A_coo.row, A_coo.col))
    A_tf = tf.sparse.SparseTensor(
        indices=indices,
        values=A_coo.data.astype(np.float32),
        dense_shape=A_coo.shape
    )
    A_tf = tf.sparse.reorder(A_tf)
    
    # 2. Extract Real Labels AND Texts
    csv_path = "../data/dataset1_tweets_combined.csv"
    print(f"Extracting true labels and text data from {csv_path}...")
    
    df = load_and_clean_data(csv_path)
    raw_labels = df['label'].values 
    
    original_texts = df['cleaned_text'].astype(str).tolist()
    
    doc_labels = tf.one_hot(raw_labels, depth=2).numpy()
    word_labels = np.zeros((num_words, 2))
    Y_matrix = np.vstack([doc_labels, word_labels])
    Y_tf = tf.convert_to_tensor(Y_matrix, dtype=tf.float32)

    # ==========================================
    # PHASE 3.5: DYNAMIC CLASS WEIGHTING
    # ==========================================
    print("\n[ANTI-BIAS PROTOCOL] Checking Class Weights Configuration...")
    if ENABLE_CLASS_WEIGHTS:
        unique_classes = np.unique(raw_labels)
        calculated_weights = compute_class_weight('balanced', classes=unique_classes, y=raw_labels)
        
        if MANUAL_MINORITY_WEIGHT is not None:
            calculated_weights[1] = MANUAL_MINORITY_WEIGHT
            
        class_weights_tf = tf.convert_to_tensor(calculated_weights, dtype=tf.float32)
        print(f"-> Class Weights ENABLED: [Class 0: {calculated_weights[0]:.4f}, Class 1: {calculated_weights[1]:.4f}]")
    else:
        # If disabled, feed 1.0 to both classes (mathematically neutral)
        class_weights_tf = tf.convert_to_tensor([1.0, 1.0], dtype=tf.float32)
        print("-> Class Weights DISABLED: Model will use standard 1.0 weight for all classes.")

    # ==========================================
    # PHASE 4: 5-FOLD CROSS-VALIDATION
    # ==========================================
    print("\nExecuting Phase 4: Initializing 5-Fold Splits...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accs, fold_precs, fold_recs, fold_f1s, fold_train_accs = [], [], [], [], []
    total_cm = np.zeros((2, 2), dtype=int)
    
    false_negatives_list = []
    
    checkpoint_dir = "../checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_model.weights.h5")

    with tf.device('/CPU:0'):
        for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(num_docs))):
            print(f"\n==================================================")
            print(f"              STARTING FOLD {fold + 1} OF 5")
            print(f"==================================================")
            
            tf.keras.backend.clear_session()
            gc.collect()

            train_mask = np.zeros(total_nodes, dtype=bool)
            test_mask = np.zeros(total_nodes, dtype=bool)
            train_mask[train_idx] = True
            test_mask[test_idx] = True
            
            train_mask_tf = tf.convert_to_tensor(train_mask)
            test_mask_tf = tf.convert_to_tensor(test_mask)
            
            model = TextGCNModel(num_classes=2, hidden_dim=64, dropout_rate=0.5, use_third_layer=False)
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.0) 

            epochs = 500
            best_test_acc = 0.0
            patience = 500
            patience_counter = 0
            
            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    predictions = model([X_tf, A_tf], training=True)
                    loss = masked_loss(Y_tf, predictions, train_mask_tf, class_weights_tf)
                    
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
            
            model.load_weights(checkpoint_path)
            final_preds_probs = model([X_tf, A_tf], training=False)
            
            train_mask_indices = np.where(train_mask)[0]
            y_true_train = np.argmax(Y_matrix[train_mask_indices], axis=1)
            y_pred_train = np.argmax(final_preds_probs.numpy()[train_mask_indices], axis=1)
            final_train_acc = accuracy_score(y_true_train, y_pred_train)
            fold_train_accs.append(final_train_acc)

            test_mask_indices = np.where(test_mask)[0]
            y_true_test = np.argmax(Y_matrix[test_mask_indices], axis=1)
            y_pred_test = np.argmax(final_preds_probs.numpy()[test_mask_indices], axis=1)
            
            for i, doc_id in enumerate(test_mask_indices):
                true_label = y_true_test[i]
                pred_label = y_pred_test[i]
                
                if true_label == 1 and pred_label == 0:
                    false_negatives_list.append({
                        "Fold": fold + 1,
                        "Doc_ID": doc_id,
                        "Tweet_Text": original_texts[doc_id]
                    })
            
            cm = confusion_matrix(y_true_test, y_pred_test, labels=[0, 1])
            total_cm += cm
            
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
    
    tn, fp, fn, tp = total_cm.ravel()
    
    print("\n--- ABSOLUTE CLASSIFICATION COUNTS ---")
    print(f"Total Undepressed Patients (Class 0): {tn + fp}")
    print(f"  -> Guessed Right (True Negative):  {tn}")
    print(f"  -> Guessed Wrong (False Positive): {fp} ")
    
    print(f"\nTotal Depressed Patients (Class 1): {fn + tp}")
    print(f"  -> Guessed Right (True Positive):  {tp} ")
    print(f"  -> Guessed Wrong (False Negative): {fn} ")
    print("==================================================")
    
    print("\n[SAVING ERROR ANALYSIS]")
    error_df = pd.DataFrame(false_negatives_list)
    error_path = "../data/error_analysis_false_negatives.csv"
    error_df.to_csv(error_path, index=False)
    print(f"Successfully saved all {len(error_df)} False Negatives to: {error_path}")
    
    print("\n--- SNEAK PEEK: 5 Tweets the Model Missed ---")
    preview_count = min(5, len(error_df))
    for idx in range(preview_count):
        print(f"Missed Tweet #{idx+1}: {error_df.iloc[idx]['Tweet_Text']}")

if __name__ == "__main__":
    main()