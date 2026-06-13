import matplotlib.pyplot as plt
import pandas as pd

def generate_thesis_results_table():
    # 1. Define the columns
    columns = ["Dataset No.", "Models", "Test Accuracy", "Precision", "Recall", "$F_1$-score"]

    # 2. Insert the data (Your exact results are included here)
    data = [
        # --- DATASET 1 ---
        ["Dataset 1", "MentalBERT", "0.7834", "0.6115", "0.5195", "0.5616"],
        ["", "MentalRoBERTa", "0.7939", "0.6252", "0.5714", "0.5965"],
        ["", "RoBERTaDepressionDetection", "0.7940", "0.6255", "0.5712", "0.5966"],
        ["", "Tri-Brain Fusion", " 0.8306 ", "0.7899", "0.7594", "0.7719"],
        
        # --- DATASET 2 ---
        ["Dataset 2", "MentalBERT", "0.7872*", "0.7987", "0.7763", "0.7857"],
        ["", "MentalRoBERTa", "0.7896", "0.7910", "0.7960", "0.7922"],
        ["", "RoBERTaDepressionDetection", "0.7869", "0.7791", "0.8074", "0.7927"],
        ["", "Tri-Brain Fusion", " 0.8690", "0.8687", "0.8701", "0.8682"],
        
        # --- DATASET 3 ---
        ["Dataset 3", "MentalBERT", "0.8279", "0.8369", "0.9258", "0.8788"],
        ["", "MentalRoBERTa", "0.9020", "0.9162", "0.9407", "0.9281"],
        ["", "RoBERTaDepressionDetection", "0.8958", "0.9037", "0.9465", "0.9245"],
        ["", "Tri-Brain Fusion ", "0.9357", "0.9302", "0.9137", "0.9210"],
        
        # --- DATASET 4 ---
        ["Dataset 4", "MentalBERT", "0.8434", "0.8832", "0.9041", "0.8935"],
        ["", "MentalRoBERTa", "0.8676", "0.9496", "0.8638", "0.9043"],
        ["", "RoBERTaDepressionDetection", "0.8654", "0.9437", "0.8646", "0.9024"],
        ["", "Tri-Brain Fusion", "0.9277", "0.9176", "0.9198", "0.9186"],
        
        # --- DATASET 5 ---
        ["Dataset 5", "MentalBERT", "0.8252", "0.9238", "0.8355", "0.8772"],
        ["", "MentalRoBERTa", "0.9104", "0.9593", "0.9193", "0.9388"],
        ["", "RoBERTaDepressionDetection", "0.9096", "0.9578", "0.9197", "0.9382"],
        ["", "Tri-Brain Fusion", "0.8755 ", "0.8760", "0.8758", "0.8752"]
    ]

    df = pd.DataFrame(data, columns=columns)

    # 3. Canvas Setup
    # Widened the canvas slightly to comfortably fit the new column proportions
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.axis('off')
    ax.axis('tight')

    # 4. Generate the Table (THE FIX IS HERE)
    # colWidths forces the specific percentage width for each column (must sum to roughly 1.0)
    # [Dataset: 12%, Models: 38%, Accuracy: 12.5%, Precision: 12.5%, Recall: 12.5%, F1: 12.5%]
    table = ax.table(cellText=df.values, 
                     colLabels=df.columns, 
                     colWidths=[0.12, 0.38, 0.125, 0.125, 0.125, 0.125], 
                     loc='center', 
                     cellLoc='center')

    # 5. Styling to match the academic paper
    table.auto_set_font_size(False)
    table.set_fontsize(11) # Bumped up font size slightly for readability
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        # Header Row Styling
        if row == 0:
            cell.set_facecolor('#808080')
            cell.set_text_props(color='white', weight='bold', fontsize=12)
            cell.set_edgecolor('#A0A0A0')
        else:
            if row % 2 == 0:
                cell.set_facecolor('#F9F9F9')
            else:
                cell.set_facecolor('white')
            
            # Highlight your model row to make it pop for the reviewers
            if "Tri-Model-Fusion" in df.iloc[row-1, 1]:
                cell.set_facecolor('#E8F8F5')
                cell.set_text_props(weight='bold')
                
            cell.set_edgecolor('#D3D3D3')

        # Specific styling for the Dataset column to create the "merged" look
        if col == 0 and row > 0:
            cell.set_text_props(ha='left')
            if df.iloc[row-1, 0] == "":
                cell.visible_edges = 'LR' 
            else:
                cell.visible_edges = 'TLR'

    # 6. Save the output
    save_path = "Comparative_Results_Table.png"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    print(f"Success! Table successfully generated and saved to {save_path}")

if __name__ == "__main__":
    generate_thesis_results_table()