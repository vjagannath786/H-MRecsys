import pandas as pd
import os
import builder
import presplit
import config
import utils_data
import numpy as np
import torch
import run
import sampling
from model import ConvModel, max_margin_loss
import math
from utils import save_txt
from metrics import get_metrics_at_k

articles = pd.read_parquet(os.path.join(config.data_path,'articles.pqt'))
customers = pd.read_parquet(os.path.join(config.data_path,'customers.pqt'))
txns = pd.read_parquet(os.path.join(config.data_path,'train.pqt'))

articles = pd.get_dummies(articles, columns=['section_name','colour_group_name'])

customers['gender'] = 1


def get_random_customers(size):
    
    #tmp = np.random.randint(1371980, size=size)

    tmp = [i for i in range(size)]
    
    df = unq_cstmrs.loc[unq_cstmrs.index.isin(tmp)]
    
    del df['index']
    
    df = df.merge(txns, on='customer_id', how='left')
    
    #df = df.merge(customers[['customer_id','age']], on='customer_id', how='left')
    
    #df.drop_duplicates(inplace=True)
    
    return df




if __name__ == "__main__":

    unq_cstmrs = customers['customer_id'].reset_index()

    tmp_txns = get_random_customers(1000)

    tmp_articles = articles.loc[articles.article_id.isin(tmp_txns.article_id)]

    tmp_cstmrs = customers.loc[customers.customer_id.isin(tmp_txns.customer_id)]

    train_df, test_df = presplit.presplit_data(tmp_txns)
    
    
    data = utils_data.DataLoader(tmp_txns)

    g = builder.create_graph(data.graph_schema)

    g = utils_data.assign_graph_features(g,tmp_cstmrs,tmp_articles, data)

    print(g)

    dim_dict = {'user': g.nodes['user'].data['features'].shape[1],
                'item': g.nodes['item'].data['features'].shape[1],
                'out': config.out_dim,
                'hidden': config.hidden_dim}
    

    #print(dim_dict)

    model = ConvModel(g, config.n_layers, dim_dict, True, config.drop_out, config.aggregator_type, config.pred, config.aggregator_hetero, True)

    #print(model)

    if config.DEVICE == 'cuda':
        model = model.to(config.DEVICE)
    

    (train_graph, train_eids_dict, valid_eids_dict, subtrain_uids, valid_uids, test_uids, \
           all_iids, ground_truth_subtrain, ground_truth_valid, all_eids_dict) = sampling.train_valid_split(g, data.ground_truth_test, [('user', 'buys', 'item')],
                                 0.05,0.1,{('user', 'buys', 'item'): ('item', 'bought-by', 'user')},
                                    False)
    

    #print(train_graph)

    fixed_params = {'neighbor_sampler': config.neighbor_sampler, 'edge_batch_size':config.edge_batch_size, 
                    'node_batch_size':config.node_batch_size,'remove_train_eids':False}

    params= {'n_layers':config.n_layers,'neg_sample_size': config.neg_sample_size }

    edgeloader_train, edgeloader_valid, nodeloader_subtrain, nodeloader_valid, nodeloader_test = sampling.generate_dataloaders(g, train_graph, train_eids_dict,valid_eids_dict, subtrain_uids, valid_uids,
                                     test_uids, all_iids, fixed_params, 1, None, True, **params)

    train_eids_len = 0
    valid_eids_len = 0
    for etype in train_eids_dict.keys():
        train_eids_len += len(train_eids_dict[etype])
        valid_eids_len += len(valid_eids_dict[etype])
    num_batches_train = math.ceil(train_eids_len / fixed_params['edge_batch_size'])
    num_batches_subtrain = math.ceil(
        (len(subtrain_uids) + len(all_iids)) / fixed_params['node_batch_size']
    )
    num_batches_val_loss = math.ceil(valid_eids_len / fixed_params['edge_batch_size'])
    num_batches_val_metrics = math.ceil(
        (len(valid_uids) + len(all_iids)) / fixed_params['node_batch_size']
    )
    num_batches_test = math.ceil(
        (len(test_uids) + len(all_iids)) / fixed_params['node_batch_size']
    )

    # Run model
    hp_sentence = params
    hp_sentence.update(fixed_params)

    hp_sentence = f'{str(hp_sentence)[1: -1]} \n'
    save_txt(f'\n \n START - Hyperparameters \n{hp_sentence}', 'results.txt', "a")


    trained_model, viz, best_metrics = run.train_model(model,config.EPOCHS,num_batches_train,num_batches_val_loss, edgeloader_train,
                    edgeloader_valid, max_margin_loss, .05,1, config.cuda, config.DEVICE, torch.optim.Adam, config.learning_rate,
                    True, train_graph, g, nodeloader_valid, nodeloader_subtrain, config.k, config.out_dim, num_batches_val_metrics,
                    num_batches_subtrain, None, ground_truth_subtrain, ground_truth_valid,
                    False, 'results.txt', 0, 5, config.pred, False, embedding_layer=config.embedding_layer)


    #print(trained_model)


    # Report performance on test set
    #log.debug('Test metrics start ...')
    trained_model.eval()
    with torch.no_grad():
        embeddings = run.get_embeddings(g,
                                    config.out_dim,
                                    trained_model,
                                    nodeloader_test,
                                    num_batches_test,
                                    config.cuda,
                                    config.DEVICE,
                                    config.embedding_layer,
                                    )

        for ground_truth in [data.ground_truth_test]:
            map = get_metrics_at_k(
                embeddings,
                g,
                trained_model,
                config.out_dim,
                ground_truth,
                all_eids_dict[('user', 'buys', 'item')],
                config.k,
                False,  # Remove already bought
                False,
                config.DEVICE,
                config.pred,
                #params['use_popularity'],
                #params['weight_popularity'],
            )

            sentence = ("TEST MAP "
                        "{:.3f}% "
                        .format(map))
            #og.info(sentence)
            save_txt(sentence, config.result_filepath, mode='a')
                    














    


