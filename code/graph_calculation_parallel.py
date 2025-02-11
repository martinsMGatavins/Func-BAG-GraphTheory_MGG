import pandas as pd
import numpy as np
import networkx as nx
import bct
import time
import multiprocessing
from joblib import Parallel, delayed

#%%

def generate_community_edge_weight_matrix(graph_matrix, community_assignment):
    num_communities = len(np.unique(community_assignment))
    
    # Initialize a matrix to store the sum of edge weights between communities
    community_edge_sum = np.zeros((num_communities, num_communities))
    
    # Initialize a matrix to store the count of edges between communities
    community_edge_count = np.zeros((num_communities, num_communities))
    
    # Loop through unique community assignments
    for community_i in range(num_communities):
        for community_j in range(community_i, num_communities):
            # Find the indices of edges between community_i and community_j
            i_indices, j_indices = np.where((community_assignment == community_i+1))[0], np.where((community_assignment == community_j+1))[0]
            
            # Sum the edge weights between community_i and community_j
            edge_weights = graph_matrix[i_indices-1][:,j_indices-1]
            upper_triangle_weights = np.triu(edge_weights,k=1)
            community_edge_sum[community_i, community_j] = np.sum(upper_triangle_weights)
            
            # Count the number of edges between community_i and community_j
            community_edge_count[community_i, community_j] = np.count_nonzero(upper_triangle_weights)
    
    # Calculate the average edge weight between communities
    with np.errstate(divide="ignore",invalid="ignore"):
        average_edge_weight = np.divide(community_edge_sum,community_edge_count)
    upper_triangle_indices = np.triu_indices(average_edge_weight.shape[0])

    # Extract the elements of the upper triangle into a flat vector
    average_edge_weight_flat = average_edge_weight[upper_triangle_indices]
    
    return average_edge_weight_flat

def mean_edge_weight(adjacency_matrix, community_assignment):
    # Get unique community IDs
    unique_communities = np.unique(community_assignment)

    # Calculate mean edge weights within and outside communities
    within_community_weights = []
    outside_community_weights = []

    for community in unique_communities:
        # Nodes belonging to the current community
        community_nodes = np.where(community_assignment == community)[0]

        # Nodes outside the current community
        outside_nodes = np.where(community_assignment != community)[0]

        # Calculate mean edge weights within the community
        within_weights = adjacency_matrix[community_nodes[:, None], community_nodes].mean(axis=1)
        within_community_weights.extend(within_weights.tolist())

        # Calculate mean edge weights outside the community
        outside_weights = adjacency_matrix[community_nodes[:, None], outside_nodes].mean(axis=1)
        outside_community_weights.extend(outside_weights.tolist())

    return within_community_weights, outside_community_weights

#%%
# Iterate through the dataframe
def loop_iter(row):
    start_time = time.time()
    sub_id = row.iloc[1] # Select subject id
    edge_weights_flat = np.array(row.iloc[2:])  # Exclude the first column (subject_id)

    # Calculate the number of nodes from the length of the community partition
    num_nodes = len(community_partition)
    
    # Reshape the flattened edge weights into a 2D array (symmetric adjacency matrix)
    edge_weights = np.zeros((num_nodes, num_nodes))
    triu_indices = np.triu_indices(num_nodes, k=1)
    edge_weights[triu_indices] = edge_weights_flat
    edge_weights.T[triu_indices] = edge_weights_flat
    
    # Apply Fisher z-transform to the edge weights
    edge_weights_z = np.arctanh(edge_weights)

    # Calculate average non-zero edge weight
    mean_non_zero_weight = np.mean(edge_weights_z[np.nonzero(edge_weights_z)])
    output = [sub_id,mean_non_zero_weight]
    
    # Calculate participation coefficient
    positive_pc, negative_pc = bct.participation_coef_sign(edge_weights_z, community_partition)
    avg_pc = (positive_pc + negative_pc) / 2
    output = np.hstack((output,avg_pc))
    
    # Calculate clustering coefficient - calculate using graphblas version of clustering
    cc = bct.clustering_coef_wu_sign(edge_weights_z,coef_type="constantini")
    output = np.hstack((output,cc))

    # Calculate within-community and between-community average nodewise edge weights
    within_community_edge_weight, between_community_edge_weight = mean_edge_weight(edge_weights_z, community_partition)
    output = np.hstack((output,within_community_edge_weight))
    output = np.hstack((output,between_community_edge_weight))
    
    # Calculate network-to-network connectivity
    network_fcon = generate_community_edge_weight_matrix(edge_weights_z, community_partition)
    output = np.hstack((output,network_fcon))

    edge_weights_z_nn = edge_weights_z
    edge_weights_z_nn[edge_weights_z_nn < 0] = 0

    # Load NetworkX representation of graph
    G = nx.from_numpy_array(edge_weights_z_nn)
    
    output_nn = []
    # Checks if non-negative adjacency matrix have more than one zero eigenvalue
    if nx.is_connected(G) == False:
        return output, output_nn # Skips non-negative graph calculations if disconnected

    mean_non_zero_weight = np.mean(edge_weights_z_nn[np.nonzero(edge_weights_z_nn)])
    output_nn = [sub_id,mean_non_zero_weight]
    
    output_nn = np.hstack((output_nn,positive_pc))
    
    cc_nn = bct.clustering_coef_wu(edge_weights_z_nn)
    output_nn = np.hstack((output_nn,cc_nn))

    within_community_edge_weight, between_community_edge_weight = mean_edge_weight(edge_weights_z_nn, community_partition)
    
    output_nn = np.hstack((output_nn,within_community_edge_weight))
    output_nn = np.hstack((output_nn,between_community_edge_weight))

    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G,normalized=True)
    betweenness_values = np.array(list(betweenness.values()))
    output_nn = np.hstack((output_nn,betweenness_values))

    # Calculate nodal efficiency (inverse of average shortest path length)
    average_shortest_path_lengths = []
    for node in G.nodes:
        shortest_path_lengths = nx.single_source_shortest_path_length(G, node)
        average_shortest_path_length = sum(shortest_path_lengths.values()) / (len(G.nodes) - 1)
        average_shortest_path_lengths.append(average_shortest_path_length)
    efficiencies = 1 / np.array(average_shortest_path_lengths)
    output_nn = np.hstack((output_nn,efficiencies))

    # Calculate non-negative network-to-network FC
    network_fcon_nn = generate_community_edge_weight_matrix(edge_weights_z, community_partition)
    output_nn = np.hstack((output_nn,network_fcon_nn))

    end_time = time.time()  # Record the end time
    iteration_time = end_time - start_time
    print("Iteration took", iteration_time, "seconds")
    return output, output_nn
#%%
if __name__ == "__main__":

    # Read functional connectivity data
    fc_list = pd.read_csv("../data/hcp_d/func_fcon-parc-888x55278_filtered.csv")

    # Read Gordon333 community partition vector
    gordon333_array = np.loadtxt("../data/gordon333CommunityAffiliation.1D")
    community_partition = np.squeeze(gordon333_array)
    num_nodes = len(community_partition)
    num_networks = len(np.unique(community_partition))
    fc_list_red = fc_list.head(2)
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(delayed(loop_iter)(row) for _, row in fc_list_red.iterrows())
    sign_array = np.vstack([k[0] for k in results])
    nn_array = np.vstack([k[1] for k in results])
    M = len(community_partition)
    C = len(np.unique(community_partition))

    # Participation coefficient
    pc_cols = [f"pcoef_{i}" for i in range(1, M + 1)]
    # Clustering coefficient
    cc_cols = [f"clust_{i}" for i in range(1, M + 1)]
    # Network-to-network FC
    network_fc_cols = [f"fc_{i}_{j}" for i in range(1, C+1) for j in range(i, C+1)]
    # Within-network FC by node
    wfc_cols = [f"within_{i}" for i in range(1, M + 1)]
    # Between-network FC by node
    bfc_cols = [f"between_{i}" for i in range(1, M + 1)]
    # Betweenness centrality
    btc_cols = [f"bcentr_{i}" for i in range(1, M + 1)]
    # Node-wise efficiency
    eff_cols = [f"eff_{i}" for i in range(1, M + 1)]

    sign_cols = ["sub_id","avg_weight"] + pc_cols + cc_cols + wfc_cols + bfc_cols + network_fc_cols
    sign_df = pd.DataFrame(sign_array,columns=sign_cols)
    sign_df.to_csv("../data/brain/signed_graph_measures.csv",index=False)

    nn_cols = ["sub_id","avg_weight"] + pc_cols + cc_cols + network_fc_cols + wfc_cols + bfc_cols + btc_cols + eff_cols
    nn_df = pd.DataFrame(nn_array,columns=nn_cols)
    nn_df.to_csv("../data/brain/nn_graph_measures.csv",index=False)

# %%
