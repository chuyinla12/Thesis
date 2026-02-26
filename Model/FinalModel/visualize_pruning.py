
import os
import argparse
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from data import load_dataset
from views import build_knn_adj, prune_low_degree_edges, prune_high_ebc_edges

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_graph(adj, labels, title, save_path, seed=42):
    # Convert to numpy if tensor
    if torch.is_tensor(adj):
        adj = adj.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # Create graph from adjacency matrix
    rows, cols = np.where(adj > 0)
    edges = zip(rows.tolist(), cols.tolist())
    G = nx.Graph()
    # Add nodes explicitly to ensure we have all sampled nodes, even isolated ones
    G.add_nodes_from(range(adj.shape[0]))
    G.add_edges_from(edges)
    
    plt.figure(figsize=(12, 12))
    
    # Use spring layout with adjusted parameters to cluster connected nodes and separate others
    # k: Optimal distance between nodes. Smaller k = tighter clusters.
    # iterations: More iterations allow the simulation to settle better.
    # scale: Scale factor for positions.
    print(f"Calculating layout for {title}...")
    pos = nx.spring_layout(G, seed=seed, k=0.08, iterations=100, scale=2.0) 
    
    # Draw nodes and edges
    # Use tab10 colormap for distinct classes
    nx.draw_networkx_nodes(G, pos, node_size=30, node_color=labels, cmap=plt.cm.tab10, alpha=0.9, linewidths=0.5, edgecolors='white')
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray', width=0.5)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {title} to {save_path}")

def sample_subgraph(adj, indices):
    """Extract subgraph for the given indices"""
    if torch.is_tensor(adj):
        # PyTorch slicing
        sub_adj = adj[indices][:, indices]
        return sub_adj
    else:
        # NumPy slicing
        sub_adj = adj[indices][:, indices]
        return sub_adj

def main():
    # Settings
    data_root = r"c:\Users\Miku12\Desktop\mywork2\data"
    extracted_root = r"c:\Users\Miku12\Desktop\contrast\FinalModel2\data_extracted"
    datasets = ["citeseer"]
    
    # Pruning parameters
    knn_k = 20
    p_low_deg = 0.1
    low_deg_score = "avg"
    p_high_ebc = 0.4
    ebc_approx_k = 256
    seed = 1
    sample_size = 1000  # Number of nodes to sample
    
    ensure_dir("vis_results")
    
    # Set global seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    for ds in datasets:
        print(f"Processing {ds}...")
        try:
            # Load data
            labels, adj, features, adj_label, feature_label = load_dataset(ds, data_root, extracted_root)
            
            # Sample nodes if graph is too large
            N = features.size(0)
            if N > sample_size:
                print(f"Sampling {sample_size} nodes from {N} total nodes...")
                sample_indices = np.sort(np.random.choice(N, sample_size, replace=False))
            else:
                sample_indices = np.arange(N)
            
            # Slice labels for visualization
            labels_vis = labels[sample_indices]
            
            # 1. Build Original KNN Graph (on full data)
            print(f"Building KNN graph for {ds} (Full)...")
            adj_knn = build_knn_adj(features, k=knn_k, metric="cosine")
            
            # Extract subgraph for visualization
            adj_knn_vis = sample_subgraph(adj_knn, sample_indices)
            
            # Save Original KNN Visualization (dropout = 0.0)
            orig_filename = f"{ds}_knn_low_0.0_high_0.0_sampled.png"
            plot_graph(adj_knn_vis, labels_vis, 
                       f"{ds} - Original KNN (Low:0.0, High:0.0)", 
                       f"vis_results/{orig_filename}", seed=seed)
            
            # 2. Apply Pruning (on full data)
            print(f"Pruning {ds} (Full)...")
            # Remove low degree edges
            adj_pruned = prune_low_degree_edges(adj_knn, ratio=p_low_deg, score=low_deg_score)
            
            # Remove high EBC edges
            adj_pruned = prune_high_ebc_edges(adj_pruned, ratio=p_high_ebc, approx_k=ebc_approx_k, seed=seed)
            
            # Extract subgraph for visualization (same indices)
            adj_pruned_vis = sample_subgraph(adj_pruned, sample_indices)
            
            # Save Pruned Graph Visualization
            pruned_filename = f"{ds}_pruned_low_{p_low_deg}_high_{p_high_ebc}_sampled.png"
            plot_graph(adj_pruned_vis, labels_vis, 
                       f"{ds} - Pruned (Low:{p_low_deg}, High:{p_high_ebc})", 
                       f"vis_results/{pruned_filename}", seed=seed)
            
        except Exception as e:
            print(f"Error processing {ds}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
