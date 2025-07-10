import numpy as np
import bct
from nilearn.connectome import vec_to_sym_matrix
import pdb
import sklearn
from multiprocessing import Pool


def load_connection_matrix(file_name):
    """
    functionality: 
        load the original brain connectivity matrix from file_name and return it
    params: 
        file_name: file_name that stores the original information for the brain connectivity matrix. 
                   note that the files are located in the folder "data" and each row represents a data point/person.
    return:
        a 3-D matrix that contains the connection matrices for all the individuals. 
        In our research, the shape is (871, n, n), where 871 is the number of individuals in the dataset and n is the number of ROIs
    """
    X = np.genfromtxt(file_name, delimiter=',')
    X = vec_to_sym_matrix(X)
    return X
def transform_connection_matrix(X):
    """
    functionality: 
        transform the connection matrix to get a new matrix that contains only 0 and positive values. 
        The new matrix contains the tranformed values using sigmoid function.
        X: the return value of load_connection_matrix(file_name) 
    """
    Xnew = np.ones_like(X)
    Xnew = 1/(1+np.exp(-X))
    return Xnew
def generate_graph_features_one_person(args):
    """
    functionality:
        generate graph features for one person
    params:
        x: connectivity matrix for one person
           this is one row in the return value of transform_connection_matrix
    return:
        an ndarray that contains all the graph metrics (1 by m, where m is the number of new features)
        According to paper: Identifying patients with Alzheimer’s disease using resting-state fMRI and graph theory,
        let's calculate the following graph features first. (see the comments below)
        
    """

    index, x = args
    print("Processing index:", index)
    
    # metrics to store the graph features
    metrics = np.array([])
    
    # x is adjacent/weighted matrix, let's convert it to distance matrix first for further use
    x_distance = bct.weight_conversion(x, 'lengths')
    # calculate ci (community structure)
    # community_louvain(W, gamma=1, ci=None, B='modularity', seed=None)
    # modularity_finetune_und(W, ci=None, gamma=1, seed=None):
    # modularity_louvain_und(W, gamma=1, hierarchy=False, seed=None)
    # modularity_und(A, gamma=1, kci=None):
    #ci1, q1 = bct.modularity_louvain_und()
    
    #ci, q = bct.modularity_und(x)
    ci,q = bct.modularity_finetune_und(x, ci=None, gamma=1, seed=42)
    #pdb.set_trace()
    # five(5) local network metrics
    ## 1). degree
    degrees = bct.degrees_und(x)
    metrics = np.concatenate((metrics, degrees))
    ## 2). participation coefficient
    pc = bct.participation_coef(x,ci)
    metrics = np.concatenate((metrics, pc))
    ## 3). betweeness centrality
    betweenness = bct.betweenness_wei(x_distance)
    metrics = np.concatenate((metrics, betweenness))
    ## 4). local efficiency
    le = bct.efficiency_wei(x, local=True)
    metrics = np.concatenate((metrics, le))
    ## 5). ratio of local to global efficiency
    ge = bct.efficiency_wei(x, local=False)
    metrics = np.concatenate((metrics, le/ge)) # but what is the point?
    # four(4) global network metrics
    ## 1). average path length
    apl, ge2, ecc, radius, diameter = bct.charpath(x_distance)
    ## 2). average clustering coefficient
    acc = bct.clustering_coef_wu(x)
    acc = np.mean(acc) # why do not use all the n numbers?
    ## 3). global efficiency
    
    ## 4). small-worldness
    seed = 42
    randomX, _ = bct.randmio_und(x, 2, seed=seed)
    random_acc = bct.clustering_coef_wu(randomX)
    random_acc = np.mean(random_acc)
    L = bct.weight_conversion(randomX, 'lengths', copy=True)
    random_apl = bct.charpath(L)
    random_apl = random_apl[0]
    small_worldness = (acc/random_acc)/(apl/random_apl)
    

    metrics = np.concatenate((metrics, np.array([apl, acc, ge, small_worldness])))
    
    # the length of metrics is n*5+4, where n is the number of ROIs
    # after calculating one metric, append it the end of metrics by using metrics = np.concatenate((metrics, one_metric))
    #return  metrics
    metrics = metrics.reshape(1,-1)
    return metrics



def generate_graph_features_all(X):
    """
    Functionality:
        Generate the graph features for all the individuals using multiprocessing
    Params:
        X: Connectivity matrix for all the individuals (shape: 871*n*n where n is the number of ROIs)
    Return:
        The graph features for all the individuals (shape: 871*m, where m is the number of new graph features)
    """
    num_processes = 8  # You can adjust this based on your system's capabilities
    chunk_size = len(X) // num_processes
    # Create a list of tuples containing index and data
    X_with_index = [(index, data) for index, data in enumerate(X)]
    with Pool(processes=num_processes) as pool:
        results = pool.map(generate_graph_features_one_person, X_with_index, chunksize=chunk_size)
    all_metrics = np.concatenate(results, axis=0)

    return all_metrics

def save_graph_features(file_name, all_metrics, author, date):
    """
    functionality: 
        save the graph metrics for all the individuals in a .npy file
    params:
        file_name: the file name of the original features/connectivities
        all_metrics: the new graph features got by generate_graph_features_all() function
        author: whose method for transform_connection_matrix(), 'Ning', 'Lade', 'Sudhan', or 'Manoj'
        date: when the new features are generated. e.g. "20240308" means Mar 28, 2024
    """
    
    
    atlas_names = ['GroupICA', 'Dict', 'CC400', 'CC200', 'BASC444', 'BASC197', 'AAL', 'Power']

    for atlas_name in atlas_names:
        if atlas_name in file_name:
            atlas = atlas_name
            break

    connectivity_types = ['corr', 'part', 'tang']

    for connectivity_type in connectivity_types:
        if connectivity_type in file_name:
            connectivity = connectivity_type
            break
    
    save_name = "./data_graph/" +type_of_transformation+ atlas + "_X_" + connectivity + "_graph_" + author + "_" + date + ".npy"
   
    with open(save_name, 'wb') as f:
        np.save(f, all_metrics)


if __name__ == "__main__":
    type_of_transformation = "sigmoid"
    # load the original features
    all_files_basc= ["GroupICA_X_corr.csv", "GroupICA_X_part.csv", "GroupICA_X_tang.csv"]
    
    for file_name in all_files_basc:
        print(file_name)
        X = load_connection_matrix("./data/"+file_name)
        # deal with negative values
        Xnew = transform_connection_matrix(X)
        # generate new graph features
        Xgraph = generate_graph_features_all(Xnew)
        save_graph_features(file_name, Xgraph, 'Lade', '2024')


