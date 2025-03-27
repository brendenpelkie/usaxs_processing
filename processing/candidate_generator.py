import uuid
import numpy as np
from models.sample import create_sample, db, Sample
from flask import current_app
import json

from scipy.stats.qmc import Sobol
from scipy.stats import qmc

from typing import Optional


import torch



device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
dtype = torch.double


from botorch.models.transforms.input import Normalize
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition.objective import GenericMCObjective
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from sklearn.manifold import TSNE

from gpytorch.kernels import MaternKernel
from gpytorch.priors import GammaPrior

import time
import warnings


from botorch import fit_gpytorch_mll
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
    qUpperConfidenceBound
)
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.sampling.normal import SobolQMCNormalSampler



def generate_candidate():
    """
    Generate a new candidate sample with random parameters
    
    Returns:
        dict: Information about the generated candidate
    """

    with open(current_app.config['EXPERIMENT_CONSTANTS_FP'], 'r') as f:
        constants = json.load(f)

    with open(current_app.config['PROCESSING_CONFIG_FP'], 'r') as f:
        processing_config = json.load(f)

    bo_config = processing_config['bayesian_optimization_constants']

    lower_bounds = [constants['TEOS']['minimum_volume_fraction'],
                    constants['ammonia']['minimum_volume_fraction'],
                    constants['ethanol']['minimum_volume_fraction'],
                    constants['ctab']['minimum_mass'],
                    constants['f127']['minimum_mass']]

    upper_bounds = [constants['TEOS']['maximum_volume_fraction'],
                    constants['ammonia']['maximum_volume_fraction'],
                    constants['ethanol']['maximum_volume_fraction'],
                    constants['ctab']['maximum_mass'],
                    constants['f127']['maximum_mass']]


    bounds_torch_norm = torch.tensor([(lower_bounds[0], upper_bounds[0]), (lower_bounds[1], upper_bounds[1]), (lower_bounds[2], upper_bounds[2]), (lower_bounds[3], upper_bounds[3]), (lower_bounds[4], upper_bounds[4])]).transpose(-1, -2)
    bounds_torch_opt = torch.tensor([[0, 0, 0, 0, 0], [1.0, 1.0, 1.0, 1.0, 1.0]], dtype = torch.float32)
    # Get x, y data for all samples 
    X, amplitude_distances, phase_distances, hiq_peaks = prepare_training_data()  

    y = bo_objective(amplitude_distances, phase_distances, hiq_peaks)
    # convert to torch and normalize things
    X_norm, y_norm = bo_preprocess(X, y, bounds_torch_norm)

    x_fractions = bayesian_optimize(X_norm, y_norm, bo_config['batch_size'], bo_config['num_restarts'], bo_config['raw_samples'], bo_config['kernel_nu'], bo_config['ard_num_dims'], bounds_torch_opt, bounds_torch_norm, acqf = 'qLogNEI', return_model = False, y_var_scale = None)

    # create new sample with volume fractions
    data = {
        'uuid': str(uuid.uuid4()),
        'teos_vf': float(x_fractions[0]),
        'ammonia_vf': float(x_fractions[1]),
        'ethanol_vf': float(x_fractions[2]),
        'ctab_mass': float(x_fractions[3]),
        'f127_mass': float(x_fractions[4]),
        'water_vf': float(1 - x_fractions[0] - x_fractions[1] - x_fractions[2] - x_fractions[3] - x_fractions[4]),
        'status': 'proposed'
    }

    sample = create_sample(data)




    sample.status = 'proposed'
    db.session.commit()

    return {
        "message": "New candidate proposed",
        "uuid": data['uuid'],
        "teos_vf": data['teos_vf']
    } 


def prepare_training_data():
    """
    Prepare training data from processed samples
    
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target values
    """
    # Get all samples with status 'processed'
    processed_samples = Sample.query.filter_by(status='processed').all()

    # Prepare training data
    X = []
    amplitude_distances = []
    phase_distances = []
    hiq_peaks = []
    
    for sample in processed_samples:
        # Features (input parameters)
        features = [
            sample.teos_vf,
            sample.ammonia_vf,
            sample.ethanol_vf,
            sample.ctab_mass,
            sample.f127_mass
        ]
        X.append(features)
        
        # Target (ap_distance)
        amplitude_distances.append(sample.amplitude_distance)
        phase_distances.append(sample.phase_distance)
        hiq_peaks.append(sample.hiq_peak)
    
    return np.array(X), np.array(amplitude_distances), np.array(phase_distances), np.array(hiq_peaks)



def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]



def initialize_model(train_x, train_y, y_var_scale = None, state_dict=None, nu = 5/2, ard_num_dims = None):
    # define models for objective and constraint
    kernel = MaternKernel(nu = nu, ard_num_dims = ard_num_dims)


    if y_var_scale == None:
        model_obj = SingleTaskGP(
            train_x,
            train_y,
            #train_Yvar=assumed_noise*torch.ones_like(train_y),
            input_transform=Normalize(d=train_x.shape[-1]),
            covar_module=kernel
        ).to(train_x)

    else:
        model_obj = SingleTaskGP(
            train_x,
            train_y,
            train_Yvar=y_var_scale*torch.ones_like(train_y),
            input_transform=Normalize(d=train_x.shape[-1]),
            covar_module=kernel
        ).to(train_x)

    # combine into a multi-output GP model
    mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj

def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]

def bayesian_optimize(x_train, y_train, batch_size, num_restarts, raw_samples, nu, ard_num_dims, bounds_torch_opt, bounds_torch_norm, acqf = 'qLogNEI', return_model = False, y_var_scale = None):
    ## init model
    mll_nei, model_nei = initialize_model(x_train, y_train, y_var_scale, ard_num_dims = ard_num_dims)    
    fit_mll = fit_gpytorch_mll(mll_nei)

    ## run acq opt
    # define the qEI and qNEI acquisition modules using a QMC sampler
    t_acqf = time.time()
    qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([raw_samples]))

    objective = GenericMCObjective(objective=obj_callable)
    
    # for best_f, we use the best observed noisy values as an approximation
    if acqf == 'qLogNEI':
        acqfunc = qLogNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=x_train,
            sampler=qmc_sampler,
            objective=objective,
            prune_baseline=True,
        )

    elif acqf == 'qLogEI':
        acqfunc = qLogExpectedImprovement(
            model = model_nei,
            best_f = y_train.max()[0],
            X_baseline = x_train,
            sampler = qmc_sampler,
            objective = objective,
            prune_baseline = True
        )
    
    # optimize for new candidates
    candidates, _ = optimize_acqf(
        acq_function=acqfunc,
        bounds=bounds_torch_opt,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,  # used for intialization heuristic
        #options={"batch_limit": 5, "maxiter": 200},
    )

    print(f'Optimized acqf in {time.time() - t_acqf} s')
    x_fractions = unnormalize(candidates, bounds_torch_norm)[0]
    print('x_fractions', x_fractions)

    if return_model:
        return x_fractions, model_nei, x_train, y_train, acqfunc
    else:
        return x_fractions

def bo_preprocess(X, y, bounds_torch_norm):
    """
    Get current data into good form for BO
    """


    y_data = - torch.tensor(y).reshape(-1,1)
    x_data_torch = torch.tensor(X)


    x_data_norm = normalize(x_data_torch, bounds_torch_norm)
    y_data_norm = normalize(y_data, torch.tensor([[y_data.min()], [y_data.max()]])).reshape(-1,1)

    return x_data_norm, y_data_norm

def bo_postprocess(candidates):
    batch = {}
    for i in range(len(candidates)):
        sample = {}
        uuid_val = str(uuid.uuid4())

        sample['teos_vol_frac'] = candidates[i,0]
        sample['ammonia_vol_frac'] = candidates[i,1]
        sample['water_vol_frac'] = candidates[i,2]

        batch[uuid_val] = sample

    return batch


def bo_objective(amplitude_distances, phase_distances, hiq_peaks):

    amplitude_weight = 0.2

    ap_sum = amplitude_weight * amplitude_distances + (1 - amplitude_weight) * phase_distances
    hiq_invert = 1/hiq_peaks


    return hiq_invert + ap_sum


def sobol_sample(m_samples, seed):
    """
    Generate sobol sample and set up data structure
    """

    with open(current_app.config['EXPERIMENT_CONSTANTS_FP'], 'r') as f:
        constants = json.load(f)

    lower_bounds = [constants['TEOS']['minimum_volume_fraction'],
                    constants['ammonia']['minimum_volume_fraction'],
                    constants['ethanol']['minimum_volume_fraction'],
                    constants['ctab']['minimum_mass'],
                    constants['f127']['minimum_mass']]

    upper_bounds = [constants['TEOS']['maximum_volume_fraction'],
                    constants['ammonia']['maximum_volume_fraction'],
                    constants['ethanol']['maximum_volume_fraction'],
                    constants['ctab']['maximum_mass'],
                    constants['f127']['maximum_mass']]

    sampler = Sobol(d=5, seed = seed)
    sampled_points = sampler.random_base2(m_samples)

    sampled_volume_fractions = qmc.scale(sampled_points, lower_bounds, upper_bounds)


    sobol_samples = {}
    for i in range(len(sampled_volume_fractions)):
        uuid_val = str(uuid.uuid4())
        sample = {}
        sample['teos_vol_frac'] = sampled_volume_fractions[i,0]
        sample['ammonia_vol_frac'] = sampled_volume_fractions[i,1]
        sample['ethanol_vol_frac'] = sampled_volume_fractions[i,2]
        sample['ctab_mass'] = sampled_volume_fractions[i,3]
        sample['f127_mass'] = sampled_volume_fractions[i,4]

        sobol_samples[uuid_val] = sample

    for sample in sobol_samples.values():
        data = {
            'uuid': str(uuid.uuid4()),
            'teos_vf': sample['teos_vol_frac'],
            'ammonia_vf': sample['ammonia_vol_frac'],
            'ethanol_vf': sample['ethanol_vol_frac'],
            'ctab_mass': sample['ctab_mass'],
            'f127_mass': sample['f127_mass'],
            'water_vf': 1 - sample['teos_vol_frac'] - sample['ammonia_vol_frac'] - sample['ethanol_vol_frac'] - sample['ctab_mass'] - sample['f127_mass'],
            'status': 'proposed'
        }

        sample = create_sample(data)

    return