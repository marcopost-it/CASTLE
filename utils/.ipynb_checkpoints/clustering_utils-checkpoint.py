import numpy as np
import pandas as pd
import progressbar

def compute_quality(clusterer,data,labels,w1,w2,w3):
    
    cluster_assignments = clusterer.labels_
    cluster, counts = np.unique(cluster_assignments,return_counts =True)
    dimensionality = data.shape[1]

    cluster_set = []

    counts = counts[cluster>=0]
    cluster = cluster[cluster>=0]


    for i in range(len(cluster)):

            mask = [True if assignment == cluster[i] else False for assignment in cluster_assignments]
            cluster_labels, cluster_counts = np.unique(labels[mask], return_counts=True, axis=None)
            majority_class = cluster_labels[np.argmax(cluster_counts)]
            cluster_set.append(np.array([i, majority_class, counts[i], compute_cluster_description(cluster[i], cluster_assignments, data, dimensionality)]))

    cluster_set = np.array(cluster_set)

    coverage = np.sum(cluster_set[:,2])/data.shape[0]
    first_term = w1 * coverage

    all_rules = cluster_set[:,3]

    purities = 0
    for cluster in all_rules:
        data_df = pd.DataFrame(data)
        labels_df = pd.DataFrame(labels)
        labels_df.columns = ['label']
        all_data = pd.concat([data_df,labels_df], axis=1)

        for f in range(len(cluster)):
            lb = cluster[f][0]
            ub = cluster[f][1]
            all_data = all_data[(all_data.iloc[:,f] <= ub) & (all_data.iloc[:,f] >= lb)]  

        rule_data_df = all_data[all_data.columns[:-1]]
        rule_labels_df = all_data[all_data.columns[-1]]

        rule_data = rule_data_df.to_numpy()
        rule_labels = rule_labels_df.to_numpy()

        i_s = np.ones((len(rule_data_df.index),1))

        _,purity = compute_purity(1, i_s, rule_labels)
        purities += purity

    second_term = w2/(len(cluster_set))*purities

    
    n_classes = (np.max(np.unique(labels))+1).astype(int)
    
    n_couples = 0
    IoUs = 0
    for i in range(n_classes-1):
        cluster_set_i = cluster_set[cluster_set[:,1] == i]
        cluster_set_others = cluster_set[cluster_set[:,1] > i]
        n_couples += len(cluster_set_i)*len(cluster_set_others)
    
        rules_i = cluster_set_i[:,3]
        rules_others = cluster_set_others[:,3]

        for Cz in rules_i:
            for Cw in rules_others:
                IoUs += IoU(Cz,Cw)
    mean_IoU = IoUs/n_couples if n_couples != 0 else 1
    third_term = w3*(1-mean_IoU)
                           
    return first_term + second_term + third_term

def compute_purity(cluster, cluster_assignments, targets):
    # Returns:
    # - cluster_label: majority class
    # - purity: ratio between number of instances in majority class and the total number of instances
    
    classes = np.unique(targets, return_index=False, return_inverse=False, return_counts=False, axis=None)
    mask = [True if assignment == cluster else False for assignment in cluster_assignments]
    labels, counts = np.unique(targets[mask], return_counts=True, axis=None)
    
    return labels[np.argmax(counts)], (np.max(counts) / np.sum(counts))

def compute_cluster_description(cluster, cluster_assignments, data, dimensionality):
    mask = [True if assignment == cluster else False for assignment in cluster_assignments]
    data = data[mask]
    rule = np.empty((dimensionality,2))
    for i in range(dimensionality):
            rule[i] = [np.min(data[:,i]), np.max(data[:,i])]
    return rule

def teta_k_computation(Cz,Cw):
    teta_k =  np.minimum(Cz[1],Cw[1]) - np.maximum(Cw[0],Cz[0])
    
    if teta_k < 0:
        teta_k = 0
    elif ( teta_k < np.finfo(float).eps ) and ( teta_k >= 0 ):   # teta_k == 0
        teta_k = 1
       
    return teta_k

def compute_overlap(Cz,Cw):
    tetas = []
    for k in range(Cz.shape[0]):
        tetas.append( teta_k_computation(Cz[k],Cw[k]) )
    overlap = np.prod(np.array(tetas))
    return overlap

def IoU(Cz,Cw):
    interArea = compute_overlap(Cz, Cw)
    
    #Cz_area = 1
    #for k in range(Cz.shape[0]):
    #    diff = Cz[k][1] - Cz[k][0]
    #    if diff != 0:
    #        Cz_area *= diff
    differences = np.abs(np.array(Cz)[:,0] - np.array(Cz)[:,1]) 
    differences[differences < np.finfo(float).eps] = 1
    Cz_area = np.prod( differences ) 
    
    #Cw_area = 1
    #for k in range(Cw.shape[0]):
    #    diff = Cw[k][1] - Cw[k][0]
    #    if diff != 0:
    #        Cw_area *= diff
    differences = np.abs(np.array(Cw)[:,0] - np.array(Cw)[:,1]) 
    differences[differences < np.finfo(float).eps] = 1
    Cw_area = np.prod( differences ) 
    
    return interArea / (Cz_area + Cw_area - interArea)

def is_in(x,cluster):
    for i in range(x.shape[0]):
        if ( x[i] < cluster[i][0] ) or ( x[i] > cluster[i][1] ):
            return False
    return True

def compute_coverage(cluster, data):
    p = []
    for d in data:
        p.append( is_in(d,cluster) )
    
    cov = np.array(p)
    cov = cov[cov==True].shape[0]
    
    return cov

def compute_coverage_speedup(cluster, data):
    
    data_df = pd.DataFrame(data)
    
    for f in range(len(cluster)):
        lb = cluster[f][0]
        ub = cluster[f][1]
        
        data_df = data_df[(data_df.iloc[:,f] <= ub) & (data_df.iloc[:,f] >= lb)]    
    cov = len(data_df.index)
    return cov