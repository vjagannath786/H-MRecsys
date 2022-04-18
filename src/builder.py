import pandas as pd
import dgl
import numpy as np
import torch

def create_ids(txns_data):

    # Create user ids
    cstmr_id = pd.DataFrame(txns_data['customer_id'].unique(),
                          columns=['customer_id'])
    cstmr_id['cstmr_new_id'] = cstmr_id.index


    # create item ids
    itm_id = pd.DataFrame(txns_data['article_id'].unique(),
                          columns=['article_id'])
    itm_id['itm_new_id'] = itm_id.index

    #print(itm_id['itm_new_id'].nunique())

    return cstmr_id, itm_id



def df_to_adjancency_list(txns_data_train,txns_data_test, cstmr_id, itm_id):

    adjacency_dict = {}

    txns_data_train = txns_data_train.merge(cstmr_id,
                                            how='left',
                                            on='customer_id')

    #print(txns_data_train.shape)
    txns_data_train = txns_data_train.merge(itm_id,
                                            how='left',
                                            on='article_id')

    #print(txns_data_train.shape)                                            

    txns_data_test = txns_data_test.merge(cstmr_id,
                                            how='left',
                                            on='customer_id')
    txns_data_test = txns_data_test.merge(itm_id,
                                            how='left',
                                            on='article_id')



    print(len(txns_data_train.itm_new_id.values))
    adjacency_dict.update(
            {
                'user_item_src': txns_data_train.cstmr_new_id.values,
                'user_item_dst': txns_data_train.itm_new_id.values,
            }
        )

    
    test_src = txns_data_test.cstmr_new_id.values
    test_dst = txns_data_test.itm_new_id.values
    ground_truth_test = (test_src, test_dst)
    


    return adjacency_dict, ground_truth_test


def create_graph(graph_schema):

    g = dgl.heterograph(graph_schema)

    return g


def import_features(g, customers, articles, cstmr_id, itm_id):

    feature_dict = {}

    # User
    
    customers = customers.merge(cstmr_id, how='left', on='customer_id')

    #customers.drop_duplicates(inplace=True)

    customers.age.fillna(20, inplace=True)

    #print(customers.columns)
    
    ids = customers.cstmr_new_id.values.astype('int')
    feats = np.stack((customers.age.values,
                        customers.gender.values),
                     axis=1)

    #user_feat = np.zeros((g.number_of_nodes('user'), 2))
    #user_feat[ids] = feats

    user_feat = torch.tensor(customers[['age','gender']].values.astype('float')).float()
    
    feature_dict['user_feat'] = user_feat

    #item
    articles = articles.merge(itm_id[['article_id','itm_new_id']],
                                      how='left',
                                      on='article_id')

    #articles.drop_duplicates(inplace=True)

    #print(articles.shape)                                      
    
    
    articles = articles[articles.itm_new_id < g.number_of_nodes(
        'item')]  # Only IDs that are in graph

    ids = articles.itm_new_id.values.astype(int)
    
    '''
    feats = np.stack((item_feat_df.is_junior.values,
                      item_feat_df.is_male.values,
                      item_feat_df.is_female.values,
                      item_feat_df.eco_design.values,
                      ),
                     axis=1)
    '''
    feats = articles.drop(['article_id'],axis=1).values

    item_feat = np.zeros((g.number_of_nodes('item'), feats.shape[1]))
    item_feat[ids] = feats
    item_feat = torch.tensor(item_feat).float()

    #item_feat = torch.zeros((g.number_of_nodes('item'), 4))
    

    #print(articles.drop(['article_id'],axis=1).shape)

    feature_dict['item_feat'] = item_feat

    #feature_dict['item_feat'] = torch.tensor(articles.drop(['article_id'],axis=1).values)





    

    return feature_dict

    

