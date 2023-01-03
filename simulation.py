from functools import partial
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import block_diag

def down_staircase(step, x): # R^d -> R
    return step * np.ceil(x / step).sum(axis=-1)

def cell_prototype(step, x): # R^d -> R^d
    return np.ceil(x / step) * step - step / 2

def get_rotation_matrix(deg): # 0 to 360
    # 2D rotation that "shifts" decision boundary by "deg" degrees counterclockwise (within some plane)
    rad = deg * np.pi / 180 
    return np.array([[np.cos(-rad), -np.sin(-rad)],
                     [np.sin(-rad), np.cos(-rad)]])

class Simulator(object):
    def __init__(
        self,
    ):
        pass

    def set_seed(self, seed: int):
        np.random.seed(seed)

    def _check_result_shapes(self, X, X_enc, A, T, Y, Y_obs): 
        assert X.shape[0] == X_enc.shape[0]
        assert X.shape[0] == A.shape[0]
        assert X.shape[0] == T.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == Y_obs.shape[0]
        assert A.ndim == 1 or all([d == 1 for d in A.shape[1:]])
        assert T.ndim == 1 or all([d == 1 for d in T.shape[1:]])
        assert Y.ndim == 1 or all([d == 1 for d in Y.shape[1:]])
        assert Y_obs.ndim == 1 or all([d == 1 for d in Y_obs.shape[1:]])


class TestingBiasSimulator(Simulator):
    def __init__(
        self,
        n_feats: int,
        # defining the down-staircase function
        label_step_size: float,
        # for simulating disparate censorship
        test_threshold_group0: Optional[float] = 1.0,
        test_threshold_group1: Optional[float] = 0.8, 
        # for simulating conditional shift in A P(Y|X) -> P(Y|X')
        label_threshold_group0: Optional[float] = None,
        label_threshold_group1: Optional[float] = None,
        rotation_group0: Optional[float] = 0.,
        rotation_group1: Optional[float] = 0.,
        pivot0: Optional[np.ndarray] = 0.4 * np.ones(2),  
        pivot1: Optional[np.ndarray] = 0.4 * np.ones(2),
        n_rot_dims0: Optional[int] = 2,
        n_rot_dims1: Optional[int] = 2,
        # clipping covariates
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        # for simulating covariate shift in A P(X) -> P(X')
        mu0: Optional[int] = None,
        mu1: Optional[int] = None,
        sigma: Optional[np.ndarray] = 0.05 * np.eye(2),
        # for simulating imperfect tests
        #terr0: Optional[float] = 0.,
        #terr1: Optional[float] = 0.,
        tfpr: Optional[float] = 0.,
        tfnr: Optional[float] = 0.,
        terr_group: Optional[int] = None,
        # label noise smoothing (for bounded theoretical regret)
        c: Optional[float] = 0.05,
        # numerical stability parameters
        eps: Optional[float] = 1e-8,
        seed: Optional[int] = 42,
    ):
        super(TestingBiasSimulator, self).__init__()
        self.n_feats = n_feats
        self.label_step_size = label_step_size
        
        self.min_value = min_value
        self.max_value = max_value
        self.mu0 = mu0
        self.mu1 = mu1
      
        self.sigma = sigma
        self.eps = eps
        self.c = c

        self.test_th0 = test_threshold_group0
        self.test_th1 = test_threshold_group1
        
        self.label_th0 = label_threshold_group0
        self.label_th1 = label_threshold_group1
        
        self.rot_0 = rotation_group0
        self.rot_1 = rotation_group1
        self.n_rot_dims0 = n_rot_dims0
        self.n_rot_dims1 = n_rot_dims1
        
        mats0 = [get_rotation_matrix(rotation_group0)] * (n_rot_dims0 // 2) + [np.eye(2)] * ((n_feats - n_rot_dims0) // 2)
        self.rmat_0 = block_diag(*mats0)
        mats1 = [get_rotation_matrix(rotation_group1)] * (n_rot_dims1 // 2) + [np.eye(2)] * ((n_feats - n_rot_dims1) // 2)
        self.rmat_1 = block_diag(*mats1)
            
        self.pivot0 = pivot0 # "origin" of the rotation -- should be on the decision boundary
        self.pivot1 = pivot1
        
        self.tfpr = tfpr
        self.tfnr = tfnr
        self.terr_group = terr_group
        
        self.down_staircase = partial(down_staircase, self.label_step_size)
        self.prototype_fn = partial(cell_prototype, self.label_step_size)
        
        np.random.seed(seed)

    def get_params(self):
        return {
            "n_feats": self.n_feats,
            "decision_min_step": self.label_step_size,
            "Xmin": self.min_value,
            "Xmax": self.max_value,
            "bd0": self.label_th0,
            "bd1": self.label_th1,
            "rot0": self.rot_0,
            "rot1": self.rot_1,
            "pivot0": self.pivot0,
            "pivot1": self.pivot1,
            "nr0": self.n_rot_dims0,
            "nr1": self.n_rot_dims1,
            "rmat0": self.rmat_0,
            "rmat1": self.rmat_1,
            "mu0": self.mu0,
            "mu1": self.mu1,
            "sigma": self.sigma,
            "terr_group": self.terr_group,
            "tfpr": self.tfpr,
            "tfnr": self.tfnr,
            "eps": self.eps,
            "t0": self.test_th0,
            "t1": self.test_th1,
        }
    


    def simulate(self, n: int, nbins: Optional[int] = 5):
        
        sample0 = np.random.multivariate_normal(self.mu0, self.sigma, size=n)
        sample1 = np.random.multivariate_normal(self.mu1, self.sigma, size=n)
        A = np.concatenate([np.zeros(len(sample0)), np.ones(len(sample1))])
        
        X_A0 = np.clip(sample0, self.min_value + self.eps, self.max_value - self.eps)
        X_A1 = np.clip(sample1, self.min_value + self.eps, self.max_value - self.eps)
                
        # Modeling rotational conditional shift
        # Operate on *prototypes* of points, not actual points -- to preserve perfect separation
        if self.rot_0 % 360 != 0.: # save comp costs -- don't multiply by identity matrix just for fun
            R_A0 = ((self.prototype_fn(X_A0) - self.pivot0) @ self.rmat_0.T) + self.pivot0
        else:
            R_A0 = self.prototype_fn(X_A0)
            
        if self.rot_1 % 360 != 0.:
            R_A1 = ((self.prototype_fn(X_A1 + self.eps) - self.pivot1) @ self.rmat_1.T) + self.pivot1
        else:
            R_A1 = self.prototype_fn(X_A1 + self.eps)
        
        # decision boundaries are calculated from the "rotated" prototypical scores (decision boundary transform)
        RS_A0 = self.down_staircase(R_A0)
        RS_A1 = self.down_staircase(R_A1)
        
        # testing decisions however are made from the un-rotated data
        S_A0 = self.down_staircase(X_A0)
        S_A1 = self.down_staircase(X_A1) 
        
        # translational conditional shift
        Y0 = RS_A0 > self.label_th0 + self.eps
        Y1 = RS_A1 > self.label_th1 + self.eps
        Y = np.concatenate([Y0, Y1])

        # Modeling clinician bias disparity
        hard_T0 = S_A0 > self.test_th0 + self.eps
        hard_T1 = S_A1 > self.test_th1 + self.eps
        
        # testing "label" smoothing
        T0 = hard_T0 | (np.random.rand(n) < self.c)
        T1 = hard_T1 | (np.random.rand(n) < self.c)
        T = np.concatenate([T0, T1])
        Y_obs = T * Y 
        
        # generate a mask for selecting positives/negatives to miss in a particular group
        if self.terr_group is not None:
            Y_obs_fp = np.random.rand(*Y_obs.shape) < self.tfpr
            Y_obs_fn = np.random.rand(*Y_obs.shape) < self.tfnr
            fp_flip_mask = (Y_obs_fp == 1) & (Y_obs == 0) # select false positives
            fn_flip_mask = (Y_obs_fn == 1) & (Y_obs == 1) # select false negatives
            Y_obs[fp_flip_mask] = 1 - Y_obs[fp_flip_mask]
            Y_obs[fn_flip_mask] = 1 - Y_obs[fn_flip_mask]
        
        X = np.concatenate([X_A0, X_A1])
        X_enc = self.bin_and_cat(X, nbins)
        
        self._check_result_shapes(X, X_enc, A, T, Y, Y_obs)
        return X, X_enc, A.astype(int), T.astype(int), Y.astype(int), Y_obs.astype(int)

    def bin_and_cat(self, X: np.ndarray, nbins: int):
        dummy_onehots = np.eye(nbins)
        indices = np.digitize(X, np.linspace(X.min(), X.max() + 1e-6, nbins + 1), right=False) - 1
        X_enc = dummy_onehots[indices].reshape(X.shape[0], -1)
        return X_enc