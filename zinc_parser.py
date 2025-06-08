import torch
import numpy as np
from scipy import sparse as sp
from tqdm import tqdm
import argparse
from torch_geometric.datasets import ZINC
import torch_geometric.transforms as T
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
import sys
# Import EGNN model from the original code
from models.egnn import EGNNModel

def main():
    # Set defaults using similar arguments as the original code
    parser = argparse.ArgumentParser(description='ZINC Canonicalization Test with PyTorch Geometric')
    parser.add_argument('--num_layers', type=int, default=5, help='number of message passing layers')
    parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension')
    parser.add_argument('--in_dim', type=int, default=128, help='input feature dimension')
    parser.add_argument('--coords_weight', type=float, default=3.0, help='coordinate update weight')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'silu', 'leakyrelu'], help='activation function')
    parser.add_argument('--norm', type=str, default='layer', choices=['layer', 'batch', 'none'], help='normalization type')
    parser.add_argument('--aggr', type=str, default='sum', choices=['sum', 'mean', 'max'], help='aggregation function')
    parser.add_argument('--residual', type=bool, default=False, help='use residual connections')
    parser.add_argument('--subset_size', type=int, default=100, help='number of ZINC graphs to test')
    parser.add_argument('--k_projectors', type=int, default=10, help='number of top eigenvalue projectors to use')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loading')
    global args
    args = parser.parse_args()

    
    # Set default precision
    torch.set_default_dtype(torch.float64)
    
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Calculate the edge feature dimension: k eigenvalues + k projector entries
    edge_attr_dim = 2 * args.k_projectors
    print(f"Using {args.k_projectors} top eigenvalues and their projectors as edge features.")
    

    # Now we can initialize the model with the correct output dimension
    # Initialize EGNN model with the provided EGNNModel class
    egnn_base = EGNNModel(
        num_layers=args.num_layers,
        emb_dim=args.emb_dim,
        proj_dim=args.k_projectors,
        aggr = 'add'
    ).to(device)


    # Load ZINC dataset from PyTorch Geometric
    print("Loading ZINC dataset from PyTorch Geometric...")
    transform = None
    dataset = ZINC(root='./data/ZINC', subset=True, transform=transform)

    # Limit to subset for testing
    subset_size = min(args.subset_size, len(dataset))
    dataset = dataset[:subset_size]
    
    print(f"Processing {len(dataset)} graphs from ZINC subset...")
    for data in tqdm(dataset):
        data = data.to(device)
        #try:
        # Check if the graph is connected
        if data.num_nodes == 0 or data.edge_index.size(1) == 0:
            continue
        # Process the data to account for full graph structure and 
        # Create edge structure
        edges = data.edge_index
        # Get adjacency matrix in similar format to original code
        n = data.num_nodes
        
        # Create adjacency matrix from edge_index
        row, col = edges
        A = torch.zeros((n, n), device=edges.device)
        A[row, col] = 1
        
        # Compute normalized adjacency (similar to original code)
        D_inv_sqrt = torch.diag(torch.sum(A, dim=1).clip(1) ** -0.5)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        
        # Compute eigendecomposition of A_norm (for consistency with original code)
        E, U = torch.linalg.eigh(A_norm)
        
        # Round eigenvalues to avoid numerical instability
        E = E.round(decimals=14)
        
        # Exclude the trivial eigenvector (last one for normalized adjacency)
        k_projectors = args.k_projectors
        E, U = torch.flip(E[:-1], dims=[0]), torch.flip(U[:, :-1], dims=[-1])
        E, U = E[:k_projectors], U[:, :k_projectors]
        
        # Count unique eigenvalues and their multiplicities
        _, mult = torch.unique(E, return_counts=True)
        
        # Find indices of eigenvectors with multiplicity 1
        single_ind = torch.where(mult == 1)[0]
        # multiply by the eigenvalues 
        #print(U, torch.exp(E[:k_projectors]))
        U = U @ torch.diagflat(torch.exp(E[:k_projectors]))
        data.pos = U[:, single_ind]  # Use eigenvectors as node positions
        data.eigvals = E[single_ind]  # Store eigenvalues
        data.x = torch.zeros((n, args.in_dim), dtype=torch.long,device=edges.device)  # Dummy node features
        #data.edge_attr = create_edge_features_with_k_projectors(edges, E, U, k=k_projectors)
        if data.pos.size(1) <  k_projectors:
            wrapper = torch.zeros((data.pos.size(0), k_projectors), dtype=torch.float64, device=edges.device)
            wrapper[:, :data.pos.size(1)] = data.pos
            data.pos = wrapper
        # Set the eigenvector dimension (number of columns in U)
        
        #dataset = [data]  # Wrap in a list for DataLoader
        #batch = Batch.from_data_list(dataset)
        #change to fully connectrd graph
        edge_index = []
        for i in range(n):
            for j in range(n):
                if i != j:  # Don't include self-loops
                    edge_index.append([i, j])
        
        # Convert to tensor and reshape to [2, num_edges]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        data.edge_index = to_undirected(edge_index)
        batch = data
        
        #foreard pass
        with torch.no_grad():
          n, d = batch.pos.shape
          device = batch.pos.device
          #recieve node features
          h = egnn_base(batch)
          
    print("Done \n")
    



if __name__ == "__main__":
    main()