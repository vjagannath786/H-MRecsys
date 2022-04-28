import presplit
import config
import builder
import os

class DataLoader:
    def __init__(self, txn_data):
        self.user_item_train, self.user_item_test = presplit.presplit_data(txn_data)
        self.cstmrs_id, self.item_id = builder.create_ids(self.user_item_train)

        self.adjacency_dict, self.ground_truth_test = builder.df_to_adjancency_list(self.user_item_train,self.user_item_test, self.cstmrs_id, self.item_id)


        #print(len(self.adjacency_dict['user_item_dst']),"asdfasdfasdf")
        self.graph_schema = {
            ('user', 'buys', 'item'):
                    (self.adjacency_dict['user_item_src'], self.adjacency_dict['user_item_dst']),
                ('item', 'bought-by', 'user'):
                    (self.adjacency_dict['user_item_dst'], self.adjacency_dict['user_item_src']),
        }


def assign_graph_features(g,customers, articles, data):
    
    features = builder.import_features(g, customers, articles, data.cstmrs_id, data.item_id)

    g.nodes['user'].data['features'] = features['user_feat']
    g.nodes['item'].data['features'] = features['item_feat']


    return g