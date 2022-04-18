import pandas as pd
import numpy as np
import torch.nn as nn
import torch

from utils import softmax




def apk(actual, predicted, k=12):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    # remove this case in advance
    # if not actual:
    #     return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=12):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])



def get_recs(g, 
             h, 
             model,
             embed_dim,
             k,
             user_ids,
             #already_bought_dict,
             #remove_already_bought=True,
             cuda=False,
             device=None,
             pred: str = 'cos',
             #use_popularity: bool = False,
             #weight_popularity=1
             ):
    """
    Computes K recommendation for all users, given hidden states, the model and what they already bought.
    """
    if cuda:  # model is already in cuda?
        model = model.to(device)
    print('Computing recommendations on {} users, for {} items'.format(len(user_ids), g.num_nodes('item')))
    recs = {}
    for user in user_ids:
        user_emb = h['user'][user]
        #already_bought = already_bought_dict[user]
        user_emb_rpt = torch.cat(g.num_nodes('item')*[user_emb]).reshape(-1, embed_dim)
        
        if pred == 'cos':
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            ratings = cos(user_emb_rpt, h['item'])

        elif pred == 'nn':
            cat_embed = torch.cat((user_emb_rpt, h['item']), 1)
            ratings = model.pred_fn.layer_nn(cat_embed)

        else:
            raise KeyError(f'Prediction function {pred} not recognized.')
            
        ratings_formatted = ratings.cpu().detach().numpy().reshape(g.num_nodes('item'),)
        softmax_ratings = softmax(ratings_formatted)
        
        #if use_popularity:
            
            #popularity_scores = g.ndata['popularity']['item'].numpy().reshape(g.num_nodes('item'),)
            #ratings_formatted = np.add(softmax_ratings, popularity_scores * weight_popularity)
        order = np.argsort(softmax_ratings)
        '''
        if remove_already_bought:
            order = [item for item in order if item not in already_bought]
        '''
        rec = order[:k]
        recs[user] = rec
    return recs



def create_ground_truth(users, items):
    df = pd.DataFrame()
    df['customer_id'] = users
    df['article_id'] = items

    #print(df)

    df = df.groupby(['customer_id']).agg({'article_id':list}).reset_index()

    #print(df)
    return df


def recs_to_metrics(recs, ground_truth_df):
    
    #print(recs)
    
    df = pd.DataFrame()

    df['customer_id'] = recs.keys()

    df['article_id_pred'] = recs.values()

    #print(df)

    
    #df.columns = ['customer_id', 'article_id']

    #df = df.groupby(['customer_id'])['article_id'].agg(list).reset_index()

    df = ground_truth_df.merge(df, on='customer_id', how='left')

    map  =  mapk(df['article_id'],df['article_id_pred'])

    return map



def get_metrics_at_k(h, 
                     g,
                     model,
                     embed_dim,
                     ground_truth,
                     bought_eids,
                     k,
                     remove_already_bought=True,
                     cuda=False,
                     device=None,
                     pred='cos'):
                     #use_popularity=False,
                     #weight_popularity=1):
    
    
    users, items = ground_truth
    
    user_ids = np.unique(users).tolist()

    ground_truth_df = create_ground_truth(users, items)

    recs = get_recs(g, h, model, embed_dim, k, user_ids, 
                    #  already_bought_dict,
                    #remove_already_bought, 
                    cuda, device, pred,) 
                    #use_popularity, weight_popularity)

    map = recs_to_metrics(recs, ground_truth_df)



    
    
    return map