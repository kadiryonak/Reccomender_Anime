# file: collaborative_filtering.py

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def create_user_item_matrix(reduced_ratings):
    R = csr_matrix((reduced_ratings['rating'], (reduced_ratings['user_id'], reduced_ratings['anime_id'])))
    return R

def train_svd(R, k=50):
    U, sigma, Vt = svds(R.astype(float), k=k)
    sigma = np.diag(sigma)
    return U, sigma, Vt

def predict_ratings(U, sigma, Vt):
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return pd.DataFrame(all_user_predicted_ratings)