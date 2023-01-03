from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
import os
import pickle
import sys
from typing import List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.svm import SVC
from tqdm.auto import tqdm

import simulation_stats
from simulation import TestingBiasSimulator
from metrics import cross_auc, cross_score
from utils import one_hot

ROC_CURVE_ORDER = ["ORACLE_OVERALL", "ORACLE_A0", "ORACLE_A1", "EXPERIMENTAL_OVERALL", "EXPERIMENTAL_A0", "EXPERIMENTAL_A1"]

def parallel_call(params):  # a helper for calling 'remote' instances
    class_name, state, fn_name, kwargs = params
    cls = getattr(sys.modules[__name__], class_name)  # get our class type
    instance = cls.__new__(cls)  # create a new instance without invoking __init__
    instance.__dict__ = state  # apply the passed state to the new instance
    method = getattr(instance, fn_name)  # get the requested method
    return method(**kwargs)  # expand arguments, call our method and return the result


class Experiment(object):
    def __init__(
        self,
        simulation: TestingBiasSimulator,
        model_name: str,
        experiment_name: str,
        experiments_dir: str,
        preview: Optional[List[str]] = [],
    ):
        self.simulation = simulation
        self.experiments_dir = experiments_dir
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.save_dir = os.path.join(self.experiments_dir, self.experiment_name)
        self.model_dir = os.path.join(self.save_dir, "models")
        self.date_str = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')  # unused for now
        self.preview = preview
        if os.path.isdir(self.save_dir) and not self.experiment_name.startswith("debug"): # OK to overwrite for debug experiments
            raise ValueError(f"Experiment is a duplicate: {self.save_dir} is a directory")

    def make_experiments_dir(self):
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
    def save_curves(self, obj):
        with open(os.path.join(self.save_dir, f"roc_curves.pkl"), "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    def save_csv(self, df: pd.DataFrame, fname: str):
        df.to_csv(os.path.join(self.save_dir, fname + ".csv"), quotechar="|")

    def save_model(self, model, model_name: str):
        with open(os.path.join(self.model_dir, f"{model_name}.pkl"), "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    def show_ci(self, df: pd.DataFrame):
        indices = [i for i in df.index if any([i.startswith(pre) for pre in self.preview])]
        preview_df = df.loc[indices]
        chunks = np.split(preview_df.sort_index(), len(preview_df) // 3)
        for chunk in chunks:
            prefix = os.path.commonprefix(chunk.index.tolist())
            print(f"{prefix[:-2]}: {chunk.iloc[1]:.4f} ({chunk.iloc[0]:.4f}, {chunk.iloc[2]:.4f})")
    
    def show_first(self, df: pd.DataFrame):
        cols = [c for c in df.columns if any([c.startswith(pre) for pre in self.preview])]
        print(df[cols].iloc[0].T)
            
    def reinitialize(self):
        raise NotImplementedError()

    def get_data(self, N=1000, seed: Optional[int] = None):
        if seed is not None:
            self.simulation.set_seed(seed)
        return self.simulation.simulate(N)

    def fit_experimental_model(self, data: Tuple[np.ndarray]):
        raise NotImplementedError()

    def fit_oracle(self, data: Tuple[np.ndarray]):
        raise NotImplementedError()

    def metrics(
        self,
        y,
        preds: np.ndarray,
        scores: np.ndarray,
        model_name: str,
        suffix: str,
        groups: Optional[np.ndarray] = None, # required for group-specific/cross-group overall stats
    ):
        acc = accuracy_score(y, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average='binary')
        auc = roc_auc_score(y, scores[:, 1])
        fprs, tprs, thresholds = roc_curve(y, scores[:, 1])
        all_metrics = {
            f"accuracy_{model_name}_{suffix}": acc,
            f"precision_{model_name}_{suffix}": precision,
            f"recall_{model_name}_{suffix}": recall,
            f"f1_{model_name}_{suffix}": f1,
            f"auc_{model_name}_{suffix}": auc,
            f"thresholds_{model_name}_{suffix}": thresholds,
        }
        curve = (tprs, fprs)
        if suffix == "overall": # FOR OVERALL-ONLY METRICS, SUCH AS CROSS-GROUP METRICS
            try:
                xauc_A0 = cross_auc(scores[(groups == 0) & (y == 0), 1], scores[(groups == 1) & (y == 1), 1])
                xauc_A1 = cross_auc(scores[(groups == 1) & (y == 0), 1], scores[(groups == 0) & (y == 1), 1])
                all_metrics[f"xauc_{model_name}_A0"] = xauc_A0
                all_metrics[f"xauc_{model_name}_A1"] = xauc_A1
            except NameError as e:
                print(f"Raised exception {e}. Skipping; you must pass in group membership vector to calculate xAUC.")
        return all_metrics, curve
                

    def report_performance(self, X: np.ndarray, y: np.ndarray, A: np.ndarray, model_name: str):
        if model_name == "oracle":
            exp_preds, exp_probs = self.oracle_predict(X, A)
        else:
            exp_preds, exp_probs = self.experimental_model_predict(X, A)
        all_metrics, roc_overall = self.metrics(y, exp_preds, exp_probs, model_name, "overall", groups=A)
        metrics_A0, roc_A0 = self.metrics(y[A == 0], exp_preds[A == 0], exp_probs[A == 0], model_name, "A0")
        metrics_A1, roc_A1 = self.metrics(y[A == 1], exp_preds[A == 1], exp_probs[A == 1], model_name, "A1")\
        
        return {
            **all_metrics,
            **metrics_A0,
            **metrics_A1,
        }, (roc_overall, roc_A0, roc_A1)

    def train(self, N, seed: Optional[int] = None):
        data = self.get_data(N=N, seed=seed)
        with warnings.catch_warnings():
            distribution_stats = simulation_stats.get_all_stats(data, "train")
        self.fit_experimental_model(data)
        self.fit_oracle(data)
        return distribution_stats

    def validate(self, N, val_seed: Optional[int] = None):
        val_data = self.get_data(N=N, seed=val_seed)
        with warnings.catch_warnings():
            distribution_stats = simulation_stats.get_all_stats(val_data, "val")
        _, X_enc, A, _, Y, _ = val_data
        oracle_metrics, roc_oracle = self.report_performance(X_enc, Y, A, "oracle")
        experimental_metrics, roc_experimental = self.report_performance(X_enc, Y, A, self.model_name)
        model_results = {
            **oracle_metrics,
            **experimental_metrics,
        }
        return model_results, distribution_stats, (*roc_oracle, *roc_experimental)

    def run_experiment(self, N=1000, val_N=1000, seed: Optional[int] = None, val_seed: Optional[int] = None):
        self.reinitialize(random_state=seed)
        distribution_stats = self.train(N, seed=seed)
        model_results, val_distribution_stats, roc_data = self.validate(val_N, val_seed=val_seed)
        results = {
            **model_results,
            **self.simulation.get_params(),
            **distribution_stats,
            **val_distribution_stats,
        }
        return results, roc_data

    def run_replication(self, num_workers=0, replications=1000, N=1000, val_N=10000):
        results = []
        curves = defaultdict(list)
        if num_workers == 0:
            for i in tqdm(range(replications)):
                results.append(self.run_experiment(N=N, val_N=val_N, seed=self.val_seed + i + 1, val_seed=self.val_seed))
        else:
            with Pool(num_workers) as p:
                with tqdm(total=replications) as pbar:
                    params = [{"N": N, "val_N": val_N, "seed": self.val_seed + i + 1, "val_seed": self.val_seed} for i in range(replications)]
                    for i, result in enumerate(p.imap_unordered(parallel_call, self._prepare_for_parallel("run_experiment", params))):    
                        metrics, rocs = result
                        results.append(metrics)
                        for k, curve_points in zip(ROC_CURVE_ORDER, rocs):
                            curves[k].append(curve_points) # (tprs, fprs)
                        pbar.update()
        results = pd.DataFrame(results)
        self.results = results
        return results, curves

    def _prepare_for_parallel(self, fn_name, args):
        for arg in args:
            yield [self.__class__.__name__, self.__dict__, fn_name, arg]
            
    def compute_ci(self, ci: Optional[int] = 95):
        assert 0 <= ci <= 100
        if self.results is None:
            print("Cannot compute CI without results.")
            return
        lq = 0.5 - ci / 200
        uq = 1 - lq
        ub = self.results.quantile(uq, axis=0).add_suffix(f"_q{100*uq:.2f}")
        lb = self.results.quantile(lq, axis=0).add_suffix(f"_q{100*lq:.2f}")
        median = self.results.median(axis=0, numeric_only=True).add_suffix("_q50")
        return pd.concat([lb, median, ub]).T

    def get_models(self):
        return [self.oracle, self.model]
    
    def oracle_predict(self, X: np.ndarray, A: np.ndarray):
        return self._predict(self.oracle, X, A)
    
    def experimental_model_predict(self, X: np.ndarray, A: np.ndarray):
        return self._predict(self.model, X, A)


class SVMExperiment(Experiment):
    def __init__(
        self,
        simulation: TestingBiasSimulator,
        model_name: str,
        experiment_name: str,
        experiments_dir: str,
        preview: Optional[List[str]] = [],
        reg: Optional[float] = 1.0,
        val_seed: Optional[int] = 42,
        kernel: Optional[str] = "rbf",
    ):
        self.reg = reg
        self.kernel = kernel
        self.model_name = self.__class__.__name__
        self.experiment_name = experiment_name
        self.val_seed = val_seed
        self.results = None
        self.reinitialize()
        super().__init__(simulation, model_name, experiment_name, experiments_dir, preview=preview)

    def reinitialize(self, random_state: Optional[int] = 42):
        self.oracle = self._initialize_model(42)
        self.model = self._initialize_model(42)

    def _initialize_model(self, random_state: Optional[int] = 42):
        return SVC(C=self.reg, kernel=self.kernel, probability=True, random_state=random_state)

    def fit_experimental_model(self, data: Tuple[np.ndarray]):
        _, X_enc, _, _, _, Y_obs = data
        self.model.fit(X_enc, Y_obs)
        
    def _predict(self, model, X: np.ndarray, A: np.ndarray):
        preds = model.predict(X)
        probs = model.predict_proba(X)
        return preds, probs

    def fit_oracle(self, data: Tuple[np.ndarray]):
        _, X_enc, _, _, Y, _ = data
        self.oracle.fit(X_enc, Y)

        
class RaceVaryingSVMExperiment(Experiment):
    def __init__(
        self,
        simulation: TestingBiasSimulator,
        model_name: str,
        experiment_name: str,
        experiments_dir: str,
        reg: Optional[float] = 1.0,
        val_seed: Optional[int] = 42,
        kernel: Optional[str] = "rbf",
        preview: Optional[List[str]] = [],
    ):
        self.reg = reg
        self.kernel = kernel
        self.model_name = self.__class__.__name__
        self.val_seed = val_seed
        self.results = None
        self.reinitialize()
        super().__init__(simulation, model_name, experiment_name, experiments_dir, preview=preview)

    def reinitialize(self, random_state: Optional[int] = 42):
        self.oracle0 = self._initialize_model(42)
        self.oracle1 = self._initialize_model(42)
        self.model0 = self._initialize_model(42)
        self.model1 = self._initialize_model(42)

    def _initialize_model(self, random_state: Optional[int] = 42):
        return SVC(C=self.reg, kernel=self.kernel, probability=True, random_state=random_state)

    def fit_experimental_model(self, data: Tuple[np.ndarray]):
        _, X_enc, A, _, _, Y_obs = data
        self.model0.fit(X_enc[A == 0], Y_obs[A == 0])
        self.model1.fit(X_enc[A == 1], Y_obs[A == 1])

    def fit_oracle(self, data: Tuple[np.ndarray]):
        _, X_enc, A, _, Y, _ = data
        self.oracle0.fit(X_enc[A == 0], Y[A == 0])
        self.oracle1.fit(X_enc[A == 1], Y[A == 1])
        
    def oracle_predict(self,  X: np.ndarray, A: np.ndarray):
        preds = np.zeros((A.size,)) # same shape as Y
        preds[A == 0] = self.oracle0.predict(X[A == 0])
        preds[A == 1] = self.oracle1.predict(X[A == 1])
        
        probs = np.zeros((A.size, 2))
        probs[A == 0] = self.oracle0.predict_proba(X[A == 0])
        probs[A == 1] = self.oracle1.predict_proba(X[A == 1])
        return preds, probs
    
    def experimental_model_predict(self,  X: np.ndarray, A: np.ndarray):
        preds = np.zeros((A.size,)) # same shape as Y
        preds[A == 0] = self.model0.predict(X[A == 0])
        preds[A == 1] = self.model1.predict(X[A == 1])
        
        probs = np.zeros((A.size, 2))
        probs[A == 0] = self.model0.predict_proba(X[A == 0])
        probs[A == 1] = self.model1.predict_proba(X[A == 1])
        return preds, probs
   
    def get_models(self):
        return [[self.oracle0, self.oracle1], [self.model0, self.model1]]
    

