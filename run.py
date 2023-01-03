from argparse import ArgumentParser
from datetime import datetime
import os
import pickle
from typing import List, Optional

import numpy as np
import pandas as pd

import experiment
from simulation import TestingBiasSimulator

RESULTS_DIRECTORY = "results/" 
            
def get_opts():
    psr = ArgumentParser()

    psr.add_argument("--experiments-dir", type=str, default=RESULTS_DIRECTORY)
    psr.add_argument("--experiment-cls", type=str, required=True)
    psr.add_argument("--experiment-name", type=str, required=True)

    psr.add_argument("--model-kwargs", type=str, default="")

    # simulation params
    psr.add_argument("--n-feats", type=int, default=10) 
    psr.add_argument("--threshold-min-step", type=float, default=0.2)
    
    # noise shift
    psr.add_argument("--t0", type=float, default=5.)
    psr.add_argument("--t1", type=float, default=5.)
    
    # conditional shift -- boundary threshold location, rotation (2d plane), and rotation axis (within 2d plane)
    psr.add_argument("--bd0", type=float, default=5.)
    psr.add_argument("--bd1", type=float, default=5.)
    psr.add_argument("--r0", type=float, default=0.)
    psr.add_argument("--r1", type=float, default=0.)
    psr.add_argument("--pivot0", type=float, default=np.ones(10) * 0.4) 
    psr.add_argument("--pivot1", type=float, default=np.ones(10) * 0.4) 
    psr.add_argument("--n-r0", type=int, default=2)
    psr.add_argument("--n-r1", type=int, default=2)
    
    # covariate shift
    psr.add_argument("--mu0", type=float, default=0.3)
    psr.add_argument("--mu1", type=float, default=0.5)
    psr.add_argument("--sigma-scale", type=float, default=0.1)
    psr.add_argument("--seed", type=int, default=42)

    # overall settings
    psr.add_argument("--train-size", type=int, default=1000)
    psr.add_argument("--val-size", type=int, default=10000)
    psr.add_argument("--reps", type=int, default=1000)
    psr.add_argument("--num-workers", type=int, default=32)
    args = psr.parse_args()
    return args


if __name__ == '__main__':
    args = get_opts()
    sim = TestingBiasSimulator(
        args.n_feats,
        args.threshold_min_step,  # smallest meaningful threshold size change
        label_threshold_group0=args.bd0,  # controls P[Y | A=0, X]
        label_threshold_group1=args.bd1,  # controls P[Y | A=1, X]
        # non-linear transformation of decision boundary
        rotation_group0=args.r0,
        rotation_group1=args.r1, 
        pivot0=args.pivot0,
        pivot1=args.pivot1,
        test_threshold_group0=args.t0,  # controls P[T | A=0, X]
        test_threshold_group1=args.t1,  # controls P[T | A=1, X]
        n_rot_dims0=args.n_r0,
        n_rot_dims1=args.n_r1,
        mu0=np.ones(args.n_feats) * args.mu0,  # affects P[Y | A, X]
        mu1=np.ones(args.n_feats) * args.mu1,  # affects P[Y | A, X]
        sigma=args.sigma_scale * np.eye(args.n_feats),  # affects P[Y | X]
        seed=args.seed,
        eps=0,
    )
    print(sim.get_params())
    experiment_cls = getattr(experiment, args.experiment_cls)
    model_kwargs = dict([kv.split("=") for kv in args.model_kwargs.split()])
    experiment = experiment_cls(
        sim,
        args.experiment_cls, # SVMExperiment
        args.experiment_name, # SETTING2_BASE_t0_t1
        args.experiments_dir,
        preview=["auc", "xauc"],
        **model_kwargs
    )
    experiment.make_experiments_dir()
    raw_results, raw_curves = experiment.run_replication(
        replications=args.reps,
        N=args.train_size,
        val_N=args.val_size,
        num_workers=args.num_workers,
    )
    
    distilled_results = experiment.compute_ci()
    #oracle, exp_model = experiment.get_models()
    
    experiment.save_curves(raw_curves)
    experiment.save_csv(raw_results, "full")
    #mgr.save_model(oracle, "oracle")
    
    experiment.show_ci(distilled_results)
    experiment.save_csv(distilled_results, "cis")
    
    #mgr.save_model(exp_model, args.experiment_name + "_model")
