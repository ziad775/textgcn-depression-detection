import matplotlib.pyplot as plt
import numpy as np

# 1. Dataset Labels
datasets = ['Dataset 1', 'Dataset 2', 
            'Dataset 3', 'Dataset 4', 'Dataset 5']

# ---------------------------------------------------------
# 2. ADD YOUR EXACT DATA HERE
# ---------------------------------------------------------
# Replace these 0.000 values with the actual F1-Scores from Mao and Han (2025)
mao_han_model_1 = [0.5616, 0.7857, 0.8788, 0.8935, 0.8772] # MentalBERT (Example) 
mao_han_model_2 = [0.5965, 0.7922, 0.9281, 0.9043, 0.9388] # MentalRoBERTa
mao_han_model_3 = [0.5966, 0.7927, 0.9245, 0.9024, 0.9382] # RobertaDepressionDetection

# Your Model F1-Scores (Tri-Brain TextGCN)
our_model_f1 = [0.7719, 0.8682, 0.9210, 0.9186, 0.8752]

# Rename these strings to match the actual names of Mao & Han's models
model1_name = "Baseline: MentalBERT"
model2_name = "Baseline: MentalRoBERTa"
model3_name = "Baseline: Roberta-Depression-Detection"
our_model_name = "Our Model (Tri-Brain)"
# ---------------------------------------------------------

# 3. Setup the Graph Details
x = np.arange(len(datasets))  # the label locations
width = 0.2  # thinner bars since we now have 4 per dataset

fig, ax = plt.subplots(figsize=(12, 6))

# 4. Create the Bars (Emerald Palette: Teal to Deep Green)
# 4. Create the Bars (Spotlight Palette: Gray baselines, Crimson red for your model)
rects1 = ax.bar(x - 1.5*width, mao_han_model_1, width, label=model1_name, color='#D3D3D3') # Light Gray
rects2 = ax.bar(x - 0.5*width, mao_han_model_2, width, label=model2_name, color='#A9A9A9') # Mid Gray
rects3 = ax.bar(x + 0.5*width, mao_han_model_3, width, label=model3_name, color='#696969') # Dark Gray
rects4 = ax.bar(x + 1.5*width, our_model_f1, width, label=our_model_name, color='#C0392B') # Bold Crimson Red



# 5. Add Labels, Title, and Formatting
ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1) # Set Y axis slightly higher than 1 so text fits
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)

# Move the legend outside the plot area
ax.legend(title='Models', bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)

# Add horizontal gridlines behind the bars
ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#B0B0B0')
ax.set_axisbelow(True)

# 6. Add the exact numbers on top of the bars
def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        # Format to 3 decimal places
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, rotation=90) # Rotated text for tight spaces

# Apply labels to all four sets of bars
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# 7. Clean up borders (remove top and right borders)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show the plot and save it
plt.tight_layout()
plt.savefig('f1_comparison_4bars.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()
