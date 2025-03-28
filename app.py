from flask import Flask, request, jsonify
import os
from celery import Celery
import csv
import json

from models.sample import db, Sample, create_sample
from processing.data_processor import process_samples
from processing.candidate_generator import generate_candidate, sobol_sample

app = Flask(__name__)
DATA_DIRECTORY = 'data'
BACKGROUND_DIRECTORY = 'background'
PROCESSING_CONFIG_FP = '/home/bgpelkie/Code/silica-np-synthesis/APS/systemconfig.json'
EXPERIMENT_CONSTANTS_FP = '/home/bgpelkie/Code/silica-np-synthesis/APS/Mesoporous_constants_APS.json'
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(BACKGROUND_DIRECTORY, exist_ok=True)
app.config['DATA_DIRECTORY'] = DATA_DIRECTORY
app.config['BACKGROUND_DIRECTORY'] = BACKGROUND_DIRECTORY
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['PROCESSING_CONFIG_FP'] = PROCESSING_CONFIG_FP
app.config['EXPERIMENT_CONSTANTS_FP'] = EXPERIMENT_CONSTANTS_FP
import logging


FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
logging.basicConfig(filename = 'debuglogging.log', level = logging.INFO, format = FORMAT)
logger = logging.getLogger(__name__)



# Initialize the database
db.init_app(app)

# Create the database
with app.app_context():
    db.create_all()

# Configure Celery
celery = Celery('app', broker='redis://localhost:6379/0')
celery.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0'
)

@app.route('/update_data', methods=['POST'])
def update_data():
    """
    Post data, save file in a directory, and add the sample information to a csv file with status of 'unprocessed'
    """
    # Check for data file
    if 'data_file' not in request.files:
        return jsonify({"error": "Data file is required"}), 400

    
    file = request.files['data_file']
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
    
    data_file = request.files['data_file']
    
    if data_file.filename == '':
        return jsonify({"error": "Data file must be selected"}), 400



    # Store data file for use below
    data_path = os.path.join(app.config['DATA_DIRECTORY'], data_file.filename)
    data_file.save(data_path)

    background_path = os.path.join(app.config['BACKGROUND_DIRECTORY'], data['background_filename'])

    data['background_fp'] = background_path


        # Update existing sample if it exists
    existing_sample = Sample.query.filter_by(id=data['id']).first()
    if existing_sample:
        # Update each field that exists in the data
        for key, value in data.items():
            if hasattr(existing_sample, key):
                setattr(existing_sample, key, value)
        if data_path:
            existing_sample.scattering_fp = data_path
        db.session.commit()
        return jsonify({
            "message": "Sample updated successfully",
            "file_path": data_path,
            "data": data
        })

    else:
        # Create a new sample
        sample = create_sample(data, scattering_fp=data_path)
        return jsonify({
            "message": "Sample created successfully",
            "file_path": data_path,
            "data": data
        })

@app.route('/upload_background', methods=['POST'])
def upload_background():
    """
    Upload a background file
    """
    file = request.files['file']
    file_path = os.path.join(app.config['BACKGROUND_DIRECTORY'], file.filename)
    file.save(file_path)
    return jsonify({
        "message": "Background file uploaded successfully",
        "file_path": file_path
    })

@app.route('/get_sample', methods=['POST'])
def get_sample():
    """
    Get a sample from the database
    """
    uuid_val = request.json.get('id')
    sample = Sample.query.with_entities(*Sample.__table__.columns).filter(Sample.id == uuid_val).all()

    # Convert SQLAlchemy result to list of dictionaries
    sample_list = []
    for s in sample:
        sample_dict = {}
        for column in Sample.__table__.columns:
            sample_dict[column.name] = getattr(s, column.name)
        sample_list.append(sample_dict)
    return jsonify(sample_list)

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

@celery.task(name='app.process_data_task')
def process_data_task():
    """
    Celery task to run data processing asynchronously
    """
    logging.info('starting celery task')
    with app.app_context():
        return process_samples()

@app.route('/process_data', methods=['GET'])
def process_data():
    """
    Trigger asynchronous data processing pipeline
    """
    # Start the Celery task
    logger.info('Starting data processing task from route')
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

@celery.task(name='app.propose_new_candidates_task')
def propose_new_candidates_task():
    """
    Celery task to propose new candidates for processing
    """
    with app.app_context():
        return generate_candidate()

@app.route('/propose_new_candidates', methods=['GET'])
def propose_new_candidates():
    """
    Propose new candidates for processing
    """
    task = propose_new_candidates_task.delay()
    return jsonify({
        "message": "New candidates proposed",
        "task_id": task.id
    })

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


@app.route('/generate_sobol_baseline', methods=['GET'])
def generate_sobol_baseline():
    """
    Generate sobol baseline samples
    """
    with app.app_context():
        sobol_sample(m_samples=5, seed=42)  # Generate 10 samples with seed 42
        return jsonify({'message': 'Sobol baseline samples generated'})

@app.route('/check_usaxs_status', methods = ['POST'])
def check_usaxs_status():
    """
    Check the status of the USAXS data
    """
    data = request.json

    sample_uuid = data['id']

    # Get list of files in data directory
    data_files = os.listdir(app.config['DATA_DIRECTORY'])
    
    # Check if sample_uuid appears in any filenames
    sample_found = any(sample_uuid in filename for filename in data_files)
    

    if sample_found:
        usaxs_status = 'complete'
    else:
        usaxs_status = 'incomplete'

    return jsonify({'usaxs_status': usaxs_status})

if __name__ == '__main__':
    app.run(debug=True) 
