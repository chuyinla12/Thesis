import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_model_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1.2)
    ax.set_ylim(0, 1.0)
    ax.axis('off')

    # Style configuration
    box_style = dict(boxstyle="round,pad=0.1", fc="white", ec="black", lw=1.5)
    tensor_style = dict(boxstyle="round,pad=0.1", fc="#E6F3FF", ec="#4A90E2", lw=1.5) # Light blue
    op_style = dict(boxstyle="round,pad=0.1", fc="#FFF2CC", ec="#D6B656", lw=1.5)     # Light yellow
    loss_style = dict(boxstyle="round,pad=0.1", fc="#E2F0D9", ec="#548235", lw=1.5)   # Light green

    # 1. Inputs & Augmentation
    ax.text(0.05, 0.85, "Input Graph\n$G=(X, A)$", ha="center", va="center", fontsize=12, bbox=box_style)
    
    # Arrows to views
    ax.annotate("", xy=(0.15, 0.75), xytext=(0.05, 0.82), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.15, 0.25), xytext=(0.05, 0.82), arrowprops=dict(arrowstyle="->", lw=1.5))

    # View 1 (Top) - GCA
    ax.text(0.15, 0.75, "View 1: GCA\nDrop Edge/Feat\n$(X_1, A_1)$", ha="center", va="center", fontsize=10, bbox=tensor_style)
    
    # View 2 (Bottom) - KNN
    ax.text(0.15, 0.25, "View 2: KNN\n$(X_2, A_2)$", ha="center", va="center", fontsize=10, bbox=tensor_style)

    # 2. GCN Encoder (Shared)
    # Top GCN
    rect_gcn1 = patches.FancyBboxPatch((0.25, 0.65), 0.1, 0.2, boxstyle="round,pad=0.02", fc="#E1D5E7", ec="#9673A6", lw=1.5)
    ax.add_patch(rect_gcn1)
    ax.text(0.30, 0.75, "GCN\nEncoder", ha="center", va="center", fontsize=10)
    
    # Bottom GCN
    rect_gcn2 = patches.FancyBboxPatch((0.25, 0.15), 0.1, 0.2, boxstyle="round,pad=0.02", fc="#E1D5E7", ec="#9673A6", lw=1.5)
    ax.add_patch(rect_gcn2)
    ax.text(0.30, 0.25, "GCN\nEncoder", ha="center", va="center", fontsize=10)

    # Connect Views to GCN
    ax.annotate("", xy=(0.25, 0.75), xytext=(0.20, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.25, 0.25), xytext=(0.20, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 3. Embeddings
    ax.text(0.45, 0.75, "Embedding\n$H_1$", ha="center", va="center", fontsize=10, bbox=tensor_style)
    ax.text(0.45, 0.25, "Embedding\n$H_2$", ha="center", va="center", fontsize=10, bbox=tensor_style)

    # Connect GCN to Embeddings
    ax.annotate("", xy=(0.40, 0.75), xytext=(0.35, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.40, 0.25), xytext=(0.35, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 4. Fusion Module
    ax.text(0.65, 0.5, "Fusion Layer\nWeighted Sum + Proj\n$H_{all}$", ha="center", va="center", fontsize=10, bbox=op_style)
    
    # Connect Embeddings to Fusion
    ax.annotate("", xy=(0.60, 0.52), xytext=(0.50, 0.75), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(0.60, 0.48), xytext=(0.50, 0.25), arrowprops=dict(arrowstyle="->", lw=1.5))

    # 5. Output Heads & Losses
    
    # Cluster Head (Top Right)
    ax.text(0.85, 0.5, "Cluster Head\nSoft Assignment $Q$", ha="center", va="center", fontsize=10, bbox=op_style)
    ax.annotate("", xy=(0.80, 0.5), xytext=(0.70, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))

    # Reconstruction Heads (Bottom Right)
    ax.text(0.65, 0.25, "Reconstruction\nHead", ha="center", va="center", fontsize=10, bbox=op_style)
    ax.annotate("", xy=(0.65, 0.29), xytext=(0.65, 0.45), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--")) # From Hall (conceptually) or H1/H2
    
    # 6. Loss Functions
    # Contrastive Loss (Between H1 and H2 and Hall)
    ax.text(0.55, 0.9, "Contrastive Loss\n$\mathcal{L}_{contra}$", ha="center", va="center", fontsize=10, bbox=loss_style)
    ax.annotate("", xy=(0.50, 0.80), xytext=(0.55, 0.86), arrowprops=dict(arrowstyle="->", lw=1.0, ls="dashed", color="green"))
    ax.annotate("", xy=(0.65, 0.55), xytext=(0.55, 0.86), arrowprops=dict(arrowstyle="->", lw=1.0, ls="dashed", color="green"))

    # Clustering Loss
    ax.text(0.85, 0.7, "Clustering Loss\n$\mathcal{L}_{cluster} + \mathcal{L}_{KL}$", ha="center", va="center", fontsize=10, bbox=loss_style)
    ax.annotate("", xy=(0.85, 0.55), xytext=(0.85, 0.65), arrowprops=dict(arrowstyle="->", lw=1.0, ls="dashed", color="green"))

    # Reconstruction Loss
    ax.text(0.65, 0.1, "Reconstruction Loss\n$\mathcal{L}_{recon}$", ha="center", va="center", fontsize=10, bbox=loss_style)
    ax.annotate("", xy=(0.65, 0.20), xytext=(0.65, 0.14), arrowprops=dict(arrowstyle="->", lw=1.0, ls="dashed", color="green"))

    # 7. Legend/Title
    plt.title("FinalModel Architecture Diagram", fontsize=16, y=0.95)
    
    # Save
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight')
    print("Diagram saved to model_architecture.png")

if __name__ == "__main__":
    try:
        draw_model_architecture()
    except Exception as e:
        print(f"Error drawing diagram: {e}")
