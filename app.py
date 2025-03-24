from flask import Flask, render_template, request, jsonify
import os
from flask_sqlalchemy import SQLAlchemy
from celery import Celery
import uuid
import csv
import time
import json
import numpy as np

from saxs_data_processing import io, manipulate, target_comparison, subtract, sasview_fitting

app = Flask(__name__)
DATA_DIRECTORY = 'data'
os.makedirs(DATA_DIRECTORY, exist_ok=True)
app.config['DATA_DIRECTORY'] = DATA_DIRECTORY
# Configure the SQLite database URI


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PROCESSING_CONFIG_FP'] = '/mnt/c/Users/bgpelkie/Code/silica-np-synthesis/APS/systemconfig.json'

# Initialize the database
db = SQLAlchemy(app)

# Define the model

class Sample(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    scattering_fp = db.Column(db.String(100), nullable=True) 
    teos_vf = db.Column(db.Float(), unique=False, nullable=False)
    ammonia_vf = db.Column(db.Float(), unique=False, nullable=False)
    ethanol_vf = db.Column(db.Float(), unique=False, nullable=False)
    water_vf = db.Column(db.Float(), unique=False, nullable=False)
    ctab_mass = db.Column(db.Float(), unique=False, nullable=False)
    f127_mass = db.Column(db.Float(), unique=False, nullable=False)
    sample_order = db.Column(db.Integer, unique=True, nullable=False)
    status = db.Column(db.String(20), default='unprocessed')
    ap_distance = db.Column(db.Float(), unique=False, nullable=True, default=None)

    def __repr__(self):
        return f'{self.id}, {type(self.id)}'


# Route to create the database
with app.app_context():
    db.create_all()

def create_sample(data, file_path=None):
    """
    Create a new sample in the database with auto-incrementing sample_order
    
    Args:
        data (dict): Dictionary containing sample data
        file_path (str, optional): Path to the scattering data file
        
    Returns:
        Sample: The newly created sample object
    """
    # Get the maximum sample_order value
    max_order = db.session.query(db.func.max(Sample.sample_order)).scalar()
    next_order = 1 if max_order is None else max_order + 1

    # Create a new sample
    sample = Sample(
        id=data['uuid'],
        scattering_fp=file_path,
        teos_vf=data.get('teos_vf'),
        ammonia_vf=data['ammonia_vf'], 
        ethanol_vf=data['ethanol_vf'],
        water_vf=data['water_vf'],
        ctab_mass=data['ctab_mass'],
        f127_mass=data['f127_mass'],
        sample_order=next_order,
        status=data.get('status', 'proposed')
    )
    
    db.session.add(sample)
    db.session.commit()
    return sample

@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Post data, save file in a directory, and add the sample information to a csv file with status of 'unprocessed'
    """
    # Check for file
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Check for JSON data
    try:
        data = request.form.get('data')
        data = json.loads(data)
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
    except Exception as e:
        return jsonify({"error": "Invalid JSON"}), 400

    # Save the file
    if file:
        file_path = os.path.join(app.config['DATA_DIRECTORY'], file.filename)
        file.save(file_path)

        # Update existing sample if it exists
        existing_sample = Sample.query.filter_by(id=data['uuid']).first()
        if existing_sample:
            # Update each field that exists in the data
            for key, value in data.items():
                if hasattr(existing_sample, key):
                    setattr(existing_sample, key, value)
            if file_path:
                existing_sample.scattering_fp = file_path
            db.session.commit()
            return jsonify({
                "message": "Sample updated successfully",
                "file_path": file_path,
                "data": data
            })

        else:
            # Create a new sample
            sample = create_sample(data, file_path)
            return jsonify({
                "message": "Sample created successfully",
                "file_path": file_path,
                "data": data
            })

@app.route('/get_sample', methods=['POST'])
def get_sample():
    """
    Get a sample from the database
    """
    print('gettingsample')
    # Get all columns from the Sample model
    
    uuid_val = request.json.get('uuid')
    sample = Sample.query.with_entities(*Sample.__table__.columns).filter(Sample.id == uuid_val).all()
    

    print(sample, len(sample), type(sample))

    # Convert SQLAlchemy result to list of dictionaries
    sample_list = []
    for s in sample:
        sample_dict = {}
        for column in Sample.__table__.columns:
            sample_dict[column.name] = getattr(s, column.name)
        sample_list.append(sample_dict)
    sample = sample_list
    return jsonify(sample)

@app.route('/dump_to_csv', methods=['GET'])
def dump_to_csv():
    """
    Dump the database to a csv file
    """
    # Convert all samples to a list of dictionaries
    samples = Sample.query.all()
    if not samples:
        return jsonify({"error": "No samples found"}), 404
        
    # Create a CSV file
    csv_path = os.path.join(app.config['DATA_DIRECTORY'], 'samples.csv')
    
    try:
        with open(csv_path, 'w', newline='') as csvfile:
            # Get column names from model
            fieldnames = Sample.__table__.columns.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write headers
            writer.writeheader()
            
            # Write each sample as a row
            for sample in samples:
                writer.writerow({column: getattr(sample, column) for column in fieldnames})
                
        return jsonify({
            "message": "Database exported successfully",
            "file_path": csv_path
        })
            
    except Exception as e:
        return jsonify({"error": f"Failed to export database: {str(e)}"}), 500
 

# Configure Celery
celery = Celery('app', broker='redis://localhost:6379/0')
celery.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

@celery.task(name='app.process_data_task')
def process_data_task():
    """
    Celery task to run data processing asynchronously
    """
    try:
        # Get all unprocessed samples from database
        with app.app_context():  # Need this to work with Flask SQLAlchemy
            unprocessed_samples = db.session.query(Sample).filter(Sample.status == 'unprocessed').all()
            
            if not unprocessed_samples:
                return {"message": "No unprocessed samples found"}

            # Convert Sample objects to dictionaries
            sample = []
            for s in unprocessed_samples:
                sample_dict = {}
                for column in Sample.__table__.columns:
                    sample_dict[column.name] = getattr(s, column.name)
                sample.append(sample_dict)

            print(sample)

            for samp in sample:
                print(f'Processing sample {samp["id"]}')
                time.sleep(20)
                ap_distance = 42

                # Update sample in database with ap_distance and mark as processed
                db.session.query(Sample).filter(Sample.id == samp["id"]).update({
                    "ap_distance": ap_distance,
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

    with open(app.config['PROCESSING_CONFIG_FP'], 'r') as f:
        config = json.load(f)

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



@app.route('/process_data', methods=['GET'])
def process_data():
    """
    Trigger asynchronous data processing pipeline
    """
    # Start the Celery task
    task = process_data_task.delay()
    
    return jsonify({
        "message": "Data processing started",
        "task_id": task.id
    })

@app.route('/task_status/<task_id>')
def task_status(task_id):
    """
    Get the status of a processing task
    """
    task = process_data_task.AsyncResult(task_id)
    if task.ready():
        return jsonify({
            "status": "completed",
            "result": task.get()
        })
    return jsonify({
        "status": "processing"
    })

@app.route('/propose_new_candidates', methods=['GET'])
def propose_new_candidates():
    """
    Propose new candidates for processing
    """
    # Get all samples with status 'processed'
    task = propose_new_candidates_task.delay()

    return jsonify({
        "message": "New candidates proposed",
        "task_id": task.id
    })


@celery.task(name='app.propose_new_candidates_task')
def propose_new_candidates_task():
    """
    Celery task to propose new candidates for processing
    """
    # Generate random parameters
    data = {
        'uuid': str(uuid.uuid4()),
        'teos_vf': np.random.random(),
        'ammonia_vf': np.random.random(),
        'ethanol_vf': np.random.random(),
        'water_vf': np.random.random(),
        'ctab_mass': np.random.random(),
        'f127_mass': np.random.random(),
        'status': 'proposed'
    }

    # Create new sample with proposed parameters
    with app.app_context():
        sample = create_sample(data)


    return {
        "message": "New candidate proposed",
        "uuid": data['uuid'],
        "teos_vf": data['teos_vf']
    }
    

@app.route('/get_proposed_candidates', methods=['GET'])
def get_proposed_candidates():
    """
    Get proposed candidates from database
    """
    candidates = Sample.query.filter(Sample.status == 'proposed').all()

    
    candidate_list = []
    for s in candidates:
        candidate_dict = {}
        for column in Sample.__table__.columns:
            candidate_dict[column.name] = getattr(s, column.name)
        candidate_list.append(candidate_dict)
    return jsonify(candidate_list)


if __name__ == '__main__':
    app.run(debug=True) 
