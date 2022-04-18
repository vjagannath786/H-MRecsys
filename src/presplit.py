import pandas as pd
import os
import config

def presplit_data(txn_data, test_date='2020-04-01'):


    #txn_data = pd.read_csv(os.path.join(config.data_path,_name), index_col=False)

    txn_data = txn_data[['customer_id','t_dat','article_id']]

    txn_data['t_dat'] = pd.to_datetime(txn_data['t_dat'])
    

    train_data = txn_data.loc[txn_data['t_dat'] <= pd.to_datetime(test_date)]

    test_data = txn_data.loc[txn_data['t_dat'] > pd.to_datetime(test_date)]


    return train_data, test_data