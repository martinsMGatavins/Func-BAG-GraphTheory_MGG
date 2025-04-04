import pandas as pd
import numpy as np
import networkx as nx
import bct
import time
import os

#%% Set home dir
os.chdir("..")

# Read functional connectivity data
fc_list = pd.read_csv("data/hcp_d/func_fcon-parc-888x55278_filtered.csv")

# Read Gordon333 community partition vector
gordon333_array = np.loadtxt("data/gordon333CommunityAffiliation.1D")
community_partition = np.squeeze(gordon333_array)
num_nodes = len(community_partition)
num_networks = len(np.unique(community_partition))

# Initialize lists to store nodewise results
participation_coef = np.empty((0,num_nodes+1))
clustering_coef = np.empty((0,num_nodes+1))
network_fc = np.empty((0,int((num_networks*(num_networks-1)/2)+num_networks+1)))
within_network_fc = np.empty((0,num_nodes+1))
between_network_fc = np.empty((0,num_nodes+1))
avg_edge_weight = np.empty((0,2))

participation_coef_nn = np.empty((0,num_nodes+1))
clustering_coef_nn = np.empty((0,num_nodes+1))
network_fc_nn = np.empty((0,int((num_networks*(num_networks-1)/2)+num_networks+1)))
within_network_fc_nn = np.empty((0,num_nodes+1))
between_network_fc_nn = np.empty((0,num_nodes+1))
betweenness_nn = np.empty((0,num_nodes+1))
efficiency_nn = np.empty((0,num_nodes+1))
avg_edge_weight_nn = np.empty((0,2))

print("initialized")
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

fc_list_red = fc_list.head(1)

#%%
# Iterate through the dataframe
for index, row in fc_list_red.iterrows():
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
    avg_edge_weight = np.vstack((avg_edge_weight,[sub_id,mean_non_zero_weight]))
    
    # Calculate participation coefficient
    positive_pc, negative_pc = bct.participation_coef_sign(edge_weights_z, community_partition)
    avg_pc = (positive_pc + negative_pc) / 2
    avg_pc = np.append(sub_id,avg_pc)
    participation_coef = np.vstack((participation_coef,avg_pc))
    
    # Calculate clustering coefficient - calculate using graphblas version of clustering
    cc = bct.clustering_coef_wu_sign(edge_weights_z,coef_type="costantini")
    cc = np.append(sub_id,cc)
    clustering_coef = np.vstack((clustering_coef,cc))

    # Calculate within-community and between-community average nodewise edge weights
    within_community_edge_weight, between_community_edge_weight = mean_edge_weight(edge_weights_z, community_partition)
    
    within_community_edge_weight = np.append(sub_id,within_community_edge_weight)
    within_network_fc = np.vstack((within_network_fc,within_community_edge_weight))

    between_community_edge_weight = np.append(sub_id,between_community_edge_weight)
    between_network_fc = np.vstack((between_network_fc,between_community_edge_weight))
    
    # Calculate network-to-network connectivity
    network_fcon = generate_community_edge_weight_matrix(edge_weights_z, community_partition)
    network_fcon = np.append(sub_id,network_fcon)
    network_fc = np.vstack((network_fc,network_fcon))

    edge_weights_z_nn = edge_weights_z
    edge_weights_z_nn[edge_weights_z_nn < 0] = 0

    # Load NetworkX representation of graph
    G = nx.from_numpy_array(edge_weights_z_nn)
    
    # Checks if non-negative adjacency matrix have more than one zero eigenvalue
    if nx.is_connected(G) == False:
        continue # Skips non-negative graph if disconnected

    mean_non_zero_weight = np.mean(edge_weights_z_nn[np.nonzero(edge_weights_z_nn)])
    avg_edge_weight_nn = np.vstack((avg_edge_weight_nn,[sub_id,mean_non_zero_weight]))
    
    positive_pc = np.append(sub_id,positive_pc)
    participation_coef_nn = np.vstack((participation_coef_nn,positive_pc))
    
    cc_nn = bct.clustering_coef_wu(edge_weights_z_nn)
    cc_nn = np.append(sub_id,cc_nn)
    clustering_coef_nn = np.vstack((clustering_coef_nn,cc_nn))

    within_community_edge_weight, between_community_edge_weight = mean_edge_weight(edge_weights_z_nn, community_partition)
    
    within_community_edge_weight = np.append(sub_id,within_community_edge_weight)
    within_network_fc_nn = np.vstack((within_network_fc_nn,within_community_edge_weight))

    between_community_edge_weight = np.append(sub_id,between_community_edge_weight)
    between_network_fc_nn = np.vstack((between_network_fc_nn,between_community_edge_weight))

    # Calculate betweenness centrality
    betweenness = nx.betweenness_centrality(G,normalized=True)
    betweenness_values = np.array(list(betweenness.values()))
    betweenness_values = np.append(sub_id,betweenness_values)
    betweenness_nn = np.vstack((betweenness_nn,betweenness_values))

    # Calculate nodal efficiency (inverse of average shortest path length)
    average_shortest_path_lengths = []
    for node in G.nodes:
        shortest_path_lengths = nx.single_source_shortest_path_length(G, node)
        average_shortest_path_length = sum(shortest_path_lengths.values()) / (len(G.nodes) - 1)
        average_shortest_path_lengths.append(average_shortest_path_length)
    efficiencies = 1 / np.array(average_shortest_path_lengths)
    efficiencies = np.append(sub_id,efficiencies)
    efficiency_nn = np.vstack((efficiency_nn,efficiencies))

    # Calculate non-negative network-to-network FC
    network_fcon_nn = generate_community_edge_weight_matrix(edge_weights_z, community_partition)
    network_fcon_nn = np.append(sub_id,network_fcon_nn)
    network_fc = np.vstack((network_fc,network_fcon_nn))

    end_time = time.time()  # Record the end time
    iteration_time = end_time - start_time
    print("Iteration took", iteration_time, "seconds")
#%%

# Load all of the tables with appropriately named columns as data frames
M = len(community_partition)
C = len(np.unique(community_partition))

# Participation coefficient
pc_cols = ["sub_id"] + [f"pcoef_{i}" for i in range(1, M + 1)]
participation_coef = pd.DataFrame(participation_coef, columns=pc_cols)
participation_coef.to_csv("data/brain/participation_coef.csv",index=False)

participation_coef_nn = pd.DataFrame(participation_coef_nn, columns=pc_cols)
participation_coef_nn.to_csv("data/brain/participation_coef_nn.csv",index=False)

# Clustering coefficient
cc_cols = ["sub_id"] + [f"clust_{i}" for i in range(1, M + 1)]
clustering_coef = pd.DataFrame(clustering_coef, columns=cc_cols)
clustering_coef.to_csv("data/brain/clustering_coef.csv",index=False)

clustering_coef_nn = pd.DataFrame(clustering_coef_nn, columns=cc_cols)
clustering_coef_nn.to_csv("data/brain/clustering_coef_nn.csv",index=False)

# Network-to-network FC
network_fc_cols = ["sub_id"] + [f"fc_{i}_{j}" for i in range(1, C + 1) for j in range(i, C + 1)]
network_fc = pd.DataFrame(network_fc, columns=network_fc_cols)
network_fc.to_csv("data/brain/network_fc.csv",index=False)

network_fc_nn = pd.DataFrame(network_fc_nn, columns=network_fc_cols)
network_fc_nn.to_csv("data/brain/network_fc_nn.csv",index=False)

# Within-network FC by node
wfc_cols = ["sub_id"] + [f"within_{i}" for i in range(1, M + 1)]
within_network_fc = pd.DataFrame(within_network_fc, columns=wfc_cols)
within_network_fc.to_csv("data/brain/within_network_fc.csv",index=False)

within_network_fc_nn = pd.DataFrame(within_network_fc_nn, columns=wfc_cols)
within_network_fc_nn.to_csv("data/brain/within_network_fc_nn.csv",index=False)

# Between-network FC by node
bfc_cols = ["sub_id"] + [f"between_{i}" for i in range(1, M + 1)]
between_network_fc = pd.DataFrame(between_network_fc, columns=bfc_cols)
between_network_fc.to_csv("data/brain/between_network_fc.csv",index=False)

between_network_fc_nn = pd.DataFrame(between_network_fc_nn, columns=bfc_cols)
between_network_fc_nn.to_csv("data/brain/between_network_fc_nn.csv",index=False)

# Betweenness centrality
btc_cols = ["sub_id"] + [f"bcentr_{i}" for i in range(1, M + 1)]
betweenness_nn = pd.DataFrame(betweenness_nn, columns=btc_cols)
betweenness_nn.to_csv("data/brain/betweenness_nn.csv",index=False)

# Node-wise efficiency
eff_cols = ["sub_id"] + [f"eff_{i}" for i in range(1, M + 1)]
efficiency_nn = pd.DataFrame(efficiency_nn, columns=eff_cols)
efficiency_nn.to_csv("data/brain/efficiency_nn.csv",index=False)


# %%
