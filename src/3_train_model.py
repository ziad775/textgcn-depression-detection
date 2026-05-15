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
    csv_path = "../data/dataset5_mixed.csv"
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

    # ... [This goes immediately after your SNEAK PEEK loop at the end of main()] ...
    
    from gnn_explainer import TextGCNExplainer, explain_patient_diagnosis

    print("\n==================================================")
    print("      XAI: EXPLAINING A SPECIFIC PREDICTION       ")
    print("==================================================")
    
    # 1. Load the Vocabulary to translate Word Nodes back to English
    # (Make sure you save your vectorizer.get_feature_names_out() to this path in Step 2!)
    try:
        vocab_list = np.load("../data/vocab.npy", allow_pickle=True)
    except FileNotFoundError:
        print("[WARNING] Could not find vocab.npy. Word nodes will show as raw indices.")
        vocab_list = None

    # 2. The Translator Function
    def translate_node(node_idx):
        if node_idx < num_docs:
            # It is a Document Node! Grab a preview of the actual tweet text
            tweet_preview = original_texts[node_idx][:60].replace('\n', ' ')
            return f"[Tweet {node_idx}]: \"{tweet_preview}...\""
        else:
            # It is a Word Node! Subtract num_docs to get the true vocab index
            word_idx = node_idx - num_docs
            if vocab_list is not None and word_idx < len(vocab_list):
                return f"[Clinical Word]: '{vocab_list[word_idx]}'"
            else:
                return f"[Word Index]: {word_idx}"

    # 3. Initialize the Explainer
    # We pass 'model' which contains the trained weights from the final fold
    print("Initializing Explainer on trained architecture...")
    explainer = TextGCNExplainer(trained_model=model, 
                                 num_features=X_tf.shape[1], 
                                 num_edges=len(A_tf.values))

    # 4. Pick a patient to explain (Let's use the very first False Negative from your error list!)
    if len(false_negatives_list) > 0:
        target_patient_idx = false_negatives_list[0]["Doc_ID"]
        target_class = 1 # We know it was supposed to be Depressed (Class 1)
        
        print(f"\n[XAI] Analyzing why the model missed Patient {target_patient_idx}...")
        
        edge_importance, feature_importance = explain_patient_diagnosis(
            explainer=explainer, 
            x=X_tf, 
            a_sparse=A_tf, 
            target_node_idx=target_patient_idx, 
            target_class=target_class, 
            epochs=200
        )

    print("      XAI: EXTRACTING GLOBAL CLINICAL MARKERS     ")
    print("==================================================")
    
    print("Initializing Explainer on trained architecture...")
    explainer = TextGCNExplainer(trained_model=model, 
                                 num_features=X_tf.shape[1], 
                                 num_edges=len(A_tf.values))

    # 1. Find all TRUE POSITIVES (Model correctly guessed Class 1)
    true_positives = []
    for i, doc_id in enumerate(test_mask_indices):
        if y_true_test[i] == 1 and y_pred_test[i] == 1:
            true_positives.append(doc_id)

    if len(true_positives) > 0:
        # 2. Sample 15 patients to save computation time
        # (Running 200 epochs on 1,000 patients would take hours)
        import random
        sample_size = min(15, len(true_positives))
        target_patients = random.sample(true_positives, sample_size)
        
        print(f"\n[XAI] Running Global Aggregation on {sample_size} True Positive patients...")
        
        # Array to hold the sum of all edge importances
        global_edge_importance = np.zeros(len(A_tf.values))
        
        for idx, patient_idx in enumerate(target_patients):
            print(f" -> Analyzing Patient {idx+1}/{sample_size} (Node {patient_idx})...")
            
            edge_imp, _ = explain_patient_diagnosis(
                explainer=explainer, 
                x=X_tf, 
                a_sparse=A_tf, 
                target_node_idx=patient_idx, 
                target_class=1, 
                epochs=100 # Reduced epochs slightly for faster loop
            )
            global_edge_importance += edge_imp
            
        # 3. Calculate the Average Importance across the sample
        global_edge_importance /= sample_size
        
        print("\n[GLOBAL ANALYSIS] The Top 10 Universal Markers for Depression in this Fold:")
        top_k = 10
        global_critical_edges = np.argsort(global_edge_importance)[-top_k:][::-1]

        for edge_idx in global_critical_edges:
            source_id = A_tf.indices[edge_idx][0].numpy()
            target_id = A_tf.indices[edge_idx][1].numpy()
            weight = global_edge_importance[edge_idx]
            
            source_text = translate_node(source_id)
            target_text = translate_node(target_id)
            
            print(f"-> {source_text}  <===(Avg Score: {weight:.4f})===>  {target_text}")
    else:
        print("No True Positives available in this fold for global analysis.")
    print("\n[ANALYSIS] Hunting for the strongest Cosine Similarity (Doc-Doc) Bridges:")
    
    # We only care about edges where BOTH the source and target are Documents (IDs < num_docs)
    cosine_edges = []
    for edge_idx in range(len(global_edge_importance)):
        source_id = A_tf.indices[edge_idx][0].numpy()
        target_id = A_tf.indices[edge_idx][1].numpy()
        
        # If both nodes are documents, it's a Cosine edge!
        if source_id < num_docs and target_id < num_docs:
            # Only keep the ones that actually have some mathematical weight
            if global_edge_importance[edge_idx] > 0.01: 
                cosine_edges.append((edge_idx, global_edge_importance[edge_idx]))
                
    # Sort them by highest score
    cosine_edges.sort(key=lambda x: x[1], reverse=True)
    
    # Print the Top 5 Semantic Bridges
    top_cosine = min(5, len(cosine_edges))
    for i in range(top_cosine):
        edge_idx, weight = cosine_edges[i]
        source_id = A_tf.indices[edge_idx][0].numpy()
        target_id = A_tf.indices[edge_idx][1].numpy()
        
        source_text = translate_node(source_id)
        target_text = translate_node(target_id)
        
        print(f"-> {source_text}  <===(Semantic Bridge Score: {weight:.4f})===>  {target_text}")        

if __name__ == "__main__":
    main()