import numpy as np
import dgl
import config


def train_valid_split(valid_graph: dgl.DGLHeteroGraph,ground_truth_test,
                      etypes,subtrain_size,
                      valid_size,reverse_etype,
                      remove_train_eids):
    
    np.random.seed(42)

    all_eids_dict = {}
    valid_eids_dict = {}
    train_eids_dict = {}
    valid_uids_all = []
    valid_iids_all = []

    for etype in etypes:
        all_eids = np.arange(valid_graph.number_of_edges(etype))
        valid_eids = all_eids[int(len(all_eids) * (1 - valid_size)):]
        valid_uids, valid_iids = valid_graph.find_edges(valid_eids, etype=etype)
        valid_uids_all.extend(valid_uids.tolist())
        valid_iids_all.extend(valid_iids.tolist())
        all_eids_dict[etype] = all_eids
        if (etype == ('user', 'buys', 'item')):
            valid_eids_dict[etype] = valid_eids
    ground_truth_valid = (np.array(valid_uids_all), np.array(valid_iids_all))
    valid_uids = np.array(np.unique(valid_uids_all))


    # Create partial graph
    train_graph = valid_graph.clone()
    for etype in etypes:
        if (etype == ('user', 'buys', 'item')):
            train_graph.remove_edges(valid_eids_dict[etype], etype=etype)
            train_graph.remove_edges(valid_eids_dict[etype], etype=reverse_etype[etype])
            train_eids = np.arange(train_graph.number_of_edges(etype))
            train_eids_dict[etype] = train_eids

    if remove_train_eids:
        train_graph.remove_edges(train_eids_dict[etype], etype=etype)
        train_graph.remove_edges(train_eids_dict[etype], etype=reverse_etype[etype])

    

    # Generate inference nodes for subtrain & ground truth for subtrain
    ## Choose the subsample of training set. For now, only users with purchases are included.
    train_uids, train_iids = valid_graph.find_edges(train_eids_dict[etypes[0]], etype=etypes[0])
    unique_train_uids = np.unique(train_uids)
    subtrain_uids = np.random.choice(unique_train_uids, int(len(unique_train_uids) * subtrain_size), replace=False)
    ## Fetch uids and iids of subtrain sample for all etypes
    subtrain_uids_all = []
    subtrain_iids_all = []
    for etype in train_eids_dict.keys():
        train_uids, train_iids = valid_graph.find_edges(train_eids_dict[etype], etype=etype)
        subtrain_eids = []
        for i in range(len(train_eids_dict[etype])):
            if train_uids[i].item() in subtrain_uids:
                subtrain_eids.append(train_eids_dict[etype][i].item())
        subtrain_uids, subtrain_iids = valid_graph.find_edges(subtrain_eids, etype=etype)
        subtrain_uids_all.extend(subtrain_uids.tolist())
        subtrain_iids_all.extend(subtrain_iids.tolist())
    ground_truth_subtrain = (np.array(subtrain_uids_all), np.array(subtrain_iids_all))
    subtrain_uids = np.array(np.unique(subtrain_uids_all))

    # Generate inference nodes for test
    test_uids, _ = ground_truth_test
    test_uids = np.unique(test_uids)
    all_iids = np.arange(valid_graph.num_nodes('item'))

    return train_graph, train_eids_dict, valid_eids_dict, subtrain_uids, valid_uids, test_uids, \
           all_iids, ground_truth_subtrain, ground_truth_valid, all_eids_dict


def generate_dataloaders(valid_graph,
                         train_graph,
                         train_eids_dict,
                         valid_eids_dict,
                         subtrain_uids,
                         valid_uids,
                         test_uids,
                         all_iids,
                         fixed_params,
                         num_workers,
                         all_sids=None,
                         embedding_layer: bool = True,
                         **params):



    n_layers = params['n_layers']
    if embedding_layer:
        n_layers = n_layers - 1
    if fixed_params['neighbor_sampler'] == 'full':
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    elif fixed_params['neighbor_sampler'] == 'partial':
        sampler = dgl.dataloading.MultiLayerNeighborSampler([
    {('user', 'buys', 'item'): 5,
     ('item', 'bought-by', 'user'): 4}] * 3, replace=True)
    else:
        raise KeyError('Neighbor sampler {} not recognized.'.format(fixed_params['neighbor_sampler']))

    sampler_n = dgl.dataloading.negative_sampler.Uniform(
        params['neg_sample_size']
    )

    if fixed_params['remove_train_eids']:
        '''
        edgeloader_train = dgl.dataloading.EdgeDataLoader(
            valid_graph,
            train_eids_dict,
            sampler,
            device=config.DEVICE,
            #g_sampling=train_graph,
            negative_sampler=sampler_n,
            batch_size=fixed_params.edge_batch_size,
            shuffle=True,
            drop_last=False,  # Drop last batch if non-full
            pin_memory=True,  # Helps the transfer to GPU
            num_workers=num_workers,
        )
        '''

        sampler = dgl.dataloading.as_edge_prediction_sampler(sampler, exclude='reverse_types',
                                        reverse_etypes={'buys': 'bought-by', 'bought-by': 'buys'},
                                        negative_sampler=sampler_n
                                        )
        edgeloader_train = dgl.dataloading.DataLoader(
                            valid_graph, train_eids_dict, sampler,
                            batch_size=fixed_params['edge_batch_size'], shuffle=True, drop_last=False, num_workers=num_workers)  



    else:
        edgeloader_train = dgl.dataloading.EdgeDataLoader(
            train_graph,
            train_eids_dict,
            sampler,
            device = config.DEVICE,
            exclude='reverse_types',
            reverse_etypes={'buys': 'bought-by', 'bought-by': 'buys'},
            negative_sampler=sampler_n,
            batch_size=fixed_params['edge_batch_size'],
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            num_workers=num_workers,

        )

    edgeloader_valid = dgl.dataloading.EdgeDataLoader(
        valid_graph,
        valid_eids_dict,
        sampler,
        device=config.DEVICE,
        #g_sampling=train_graph,
        negative_sampler=sampler_n,
        batch_size=fixed_params['edge_batch_size'],
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    nodeloader_subtrain = dgl.dataloading.NodeDataLoader(
        train_graph,
        {'user': subtrain_uids, 'item': all_iids},
        sampler,
        batch_size=fixed_params['node_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    nodeloader_valid = dgl.dataloading.NodeDataLoader(
        train_graph,
        {'user': valid_uids, 'item': all_iids},
        sampler,
        batch_size=fixed_params['node_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )

    test_node_ids = {'user': test_uids, 'item': all_iids}
    if 'sport' in valid_graph.ntypes:
        test_node_ids['sport'] = all_sids

    nodeloader_test = dgl.dataloading.NodeDataLoader(
        valid_graph,
        test_node_ids,
        sampler,
        batch_size=fixed_params['node_batch_size'],
        shuffle=True,
        drop_last=False,
        num_workers=num_workers
    )

    return edgeloader_train, edgeloader_valid, nodeloader_subtrain, nodeloader_valid, nodeloader_test