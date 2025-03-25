import time
from models.sample import Sample, db

from saxs_data_processing import io, manipulate, target_comparison, subtract, sasview_fitting
import json
import numpy as np
import logging

FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
logging.basicConfig(filename = 'dataprocessor.log', level = logging.INFO, format = FORMAT)
logger = logging.getLogger(__name__)


# get logger 
def process_samples():
    """
    Process all unprocessed samples in the database
    
    Returns:
        dict: Message indicating processing status
    """
    logger.info('starting process_samples helper function')
    try:

        # Get all unprocessed samples from database
        unprocessed_samples = db.session.query(Sample).filter(Sample.status == 'unprocessed').all()
        
        if not unprocessed_samples:
            return {"message": "No unprocessed samples found"}

        # Convert Sample objects to dictionaries
        samples = []
        for s in unprocessed_samples:
            sample_dict = {}
            for column in Sample.__table__.columns:
                sample_dict[column.name] = getattr(s, column.name)
            samples.append(sample_dict)

        for sample in samples:
            print(f'Processing sample {sample["id"]}')
            logger.info('Starting data processing pipeline')
            amplitude, phase, hiq_peak = process_data_pipeline(sample)

            # Update sample in database with ap_distance and mark as processed
            db.session.query(Sample).filter(Sample.id == sample["id"]).update({
                "amplitude_distance": amplitude,
                "phase_distance": phase,
                "hiq_peak": hiq_peak,
                "status": "processed"
            })
            db.session.commit()

        return {"message": "Processing completed successfully"}
    except Exception as e:
         return {"status": "error", "message": str(e)} 


def process_data_pipeline(sample):
    """
    Function to do the actual SAXS data processing and amplitude phase distance calculation
    """

    # get all the constants from config

    #TODO: Fix this to load from app context 

    logger.info('Loading config')
    with open('/home/bgpelkie/Code/silica-np-synthesis/APS/systemconfig.json', 'r') as f:
        config = json.load(f)

    logger.info('loaded config')
    config = config['processing_constants']
        
    q_min_subtract = config['q_min_subtract']
    q_max_subtract = config['q_max_subtract'] 
    q_min_spl = config['q_min_spl']
    n_interpolate_gridpts = config['n_interpolate_gridpts']
    q_split_hilo = config['q_split_hilo']
    target_r_nm = config['target_r_nm']
    target_pdi = config['target_pdi']
    sld_silica = config['sld_silica']
    sld_ethanol = config['sld_ethanol']
    savgol_n = config['savgol_n']
    savgol_order = config['savgol_order']
    min_data_len = config['min_data_len']
    spline_s = config['spline_s']
    spline_k = config['spline_k']
    scale_n_avg = config['scale_n_avg']
    apdist_optim = config['apdist_optim']
    apdist_grid_dim = config['apdist_grid_dim']

    target_r_angs = target_r_nm * 10


    # calculate target scattering profile 
    q_grid = np.linspace(np.log10(q_min_spl), np.log10(q_split_hilo), n_interpolate_gridpts)

    q_grid_nonlog = 10**q_grid
    target_I = target_comparison.target_intensities(q_grid_nonlog, target_r_angs, target_pdi, sld_silica, sld_ethanol)
    target_I = np.log10(target_I)

    # load data from disk
    data_fp = sample['scattering_fp']
    background_fp = sample['background_fp']

    data = io.read_1D_data(data_fp)
    background = io.read_1D_data(background_fp)




    # background subtraction
    subtracted = subtract.chop_subtract(data[0], background[0], hiq_thresh=1)
    subtracted = subtracted[subtracted['q'] < q_max_subtract]
    subtracted = subtracted[~subtracted['I'].isna()]
    subtracted = subtracted[subtracted['I'] > 0] # drop negative values 

    # split hi and lo q
    subtracted_loq = subtracted[subtracted['q'] < q_split_hilo]
    subtracted_hiq = subtracted[subtracted['q'] > q_split_hilo]

    # lo-q apdist fitting pipeline
    q_log = np.log10(subtracted_loq['q'].to_numpy())
    I_log = np.log10(subtracted_loq['I'].to_numpy())

    I_savgol = manipulate.denoise_intensity(I_log, savgol_n = savgol_n, savgol_order = savgol_order)
    I_spline = manipulate.fit_interpolate_spline(q_log, I_savgol, q_grid, s = spline_s, k = spline_k)

    # scale onto target 
    I_scaled = manipulate.scale_intensity_highqavg(I_spline, target_I, n_avg = scale_n_avg)


    # amplitude phase distance calculation
    amplitude, phase = target_comparison.ap_distance(q_grid, I_scaled, target_I, optim = apdist_optim, grid_dim = apdist_grid_dim)

    # integrate hi-q data
    hiq_peak = np.trapezoid(subtracted_hiq['I'], x = subtracted_hiq['q'])


    return amplitude, phase, hiq_peak