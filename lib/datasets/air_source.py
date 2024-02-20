import os

import numpy as np
import pandas as pd

from lib import datasets_path
from .pd_dataset import PandasDataset
from ..utils.utils import disjoint_months, infer_mask, compute_mean, geographical_distance, thresholded_gaussian_kernel
from ..utils import sample_mask, sample_mask_block

class AirSource(PandasDataset):
    SEED = 3210
    def __init__(self, impute_nans=False, small=False, freq='60T', masked_sensors=None):
        self.random = np.random.default_rng(self.SEED)
        self.test_months = [3, 6, 9, 12]
        self.eval_mask = None
        df, mask, dist, df_raw= self.load()
        self.df_raw = df_raw

        self.dist = dist
        if masked_sensors is None:
            self.masked_sensors = list()
        else:
            self.masked_sensors = list(masked_sensors)
        super().__init__(dataframe=df, u=None, mask=mask, name='air', freq=freq, aggr='nearest')
        
    def load(self):
        eval_mask = None
        df = pd.read_csv('./datasets/air_quality/tianjin.csv',index_col=0)
        df.index = pd.to_datetime(df.index)
        df_raw = df
        # stations = pd.DataFrame(pd.read_hdf(path, 'stations'))
        mask = ~np.isnan(df.values)
        df.fillna(method='ffill', axis=0, inplace=True)
        stations = pd.read_csv('./datasets/air_quality/station_t.csv',index_col=0)
        # compute distances from latitude and longitude degrees
        st_coord = stations.loc[:, ['latitude', 'longitude']]
        dist = geographical_distance(st_coord, to_rad=True).values
        return df, mask, dist, df_raw
    
    # def load(self, impute_nans=True, small=False, masked_sensors=None):
    #     # load readings and stations metadata
    #     df, stations, eval_mask = self.load_raw(small)
    #     # compute the masks
    #     mask = (~np.isnan(df.values)).astype('uint8')  # 1 if value is not nan else 0
    #     if eval_mask is None:
    #         eval_mask = infer_mask(df, infer_from=self.infer_eval_from)

    #     eval_mask = eval_mask.values.astype('uint8')
    #     if masked_sensors is not None:
    #         eval_mask[:, masked_sensors] = np.where(mask[:, masked_sensors], 1, 0)
    #     self.eval_mask = eval_mask  # 1 if value is ground-truth for imputation else 0
    #     # eventually replace nans with weekly mean by hour
    #     if impute_nans:
    #         df = df.fillna(compute_mean(df))
    #     # compute distances from latitude and longitude degrees
    #     st_coord = stations.loc[:, ['latitude', 'longitude']]
    #     dist = geographical_distance(st_coord, to_rad=True).values
    #     return df, dist, mask

    @property
    def mask(self):
        if self._mask is None:
            return self.df.values != 0.
        return self._mask

    def get_similarity(self, thr=0.1, include_self=False, force_symmetric=False, sparse=False, **kwargs):
        theta = np.std(self.dist[:27, :27])  # use same theta for both air and air36
        adj = thresholded_gaussian_kernel(self.dist, theta=theta, threshold=thr)
        if not include_self:
            adj[np.diag_indices_from(adj)] = 0.
        if force_symmetric:
            adj = np.maximum.reduce([adj, adj.T])
        if sparse:
            import scipy.sparse as sps
            adj = sps.coo_matrix(adj)
      
        return adj



class MissingAirSource(AirSource):
    SEED = 56789
    def __init__(self, p_fault=0.0015, p_noise=0.05, fixed_mask=False):
        super(MissingAirSource, self).__init__()
        self.rng = np.random.default_rng(self.SEED)
        self.p_fault = p_fault
        self.p_noise = p_noise
        self.fixed_mask = fixed_mask

        eval_mask = sample_mask(self.mask[0:7260,:], p=self.p_noise)
        eval_mask_block = sample_mask_block(self.mask[-1500:,:].shape,
                                self.p_fault,
                                self.p_noise,
                                min_seq=5,
                                max_seq=15,
                                rng=self.rng)
        if self.fixed_mask == False:
            self.eval_mask = np.concatenate((eval_mask,eval_mask_block),axis=0)
            np.save('./datasets/air_quality/beijing_mask.npy', self.eval_mask)
        else:
            self.eval_mask = np.load('./datasets/air_quality/beijing_mask.npy')
        # self.eval_mask = eval_mask
      
    @property
    def training_mask(self):
        # print(type(self.mask))
        # print(self.mask.size - np.count_nonzero(self.mask))
   
        return self.mask if self.eval_mask is None else (self.mask & (1 - self.eval_mask))

    def splitter(self, dataset, val_len=0, test_len=0, window=0):
        idx = np.arange(len(dataset))
    
        test_len = 1000
        val_len = 500

        test_start = len(idx) - test_len
        val_start = test_start - val_len


        return [idx[:val_start - window], idx[val_start:test_start - window], idx[test_start:]]

