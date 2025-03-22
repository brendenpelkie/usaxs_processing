from flask import Flask, render_template, request, jsonify
import os
from flask_sqlalchemy import SQLAlchemy
from celery import Celery
import uuid
import csv

import json

app = Flask(__name__)
DATA_DIRECTORY = 'data'
os.makedirs(DATA_DIRECTORY, exist_ok=True)
app.config['DATA_DIRECTORY'] = DATA_DIRECTORY
# Configure the SQLite database URI


app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize the database
db = SQLAlchemy(app)

# Define the model

class Sample(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    scattering_fp = db.Column(db.String(100), nullable=True) 
    teos_vf = db.Column(db.Float(), unique=False, nullable=False)
    status = db.Column(db.String(20), default='unprocessed')
    ap_distance = db.Column(db.Float(), unique=False, nullable=True, default=None)

    def __repr__(self):
        return f'{self.id}, {type(self.id)}'

# class Sample(db.Model):
#     id = db.Column(db.Uuid(), primary_key=True)
#     scattering_fp = db.Column(db.String(100), nullable=True)
#     teos_vf = db.Column(db.Float(), unique=False, nullable=False)

#    def __repr__(self):
#         return f'{self.id}, {type(self.id)}'

# Route to create the database
with app.app_context():
    db.create_all()


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

        # Add sample information to the database
        print('uuid string: ', data['uuid'], type(data['uuid']))
        print('UUID: ', uuid.UUID(data['uuid']))
        sample = Sample(
            id=data['uuid'],
            scattering_fp=file_path,
            teos_vf=data['teos_vf']
        )
        print(sample)

        db.session.add(sample)
        db.session.commit()
        return jsonify({
            "message": "File uploaded successfully",
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
    sample = Sample.query.with_entities(*Sample.__table__.columns).all()
    

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
 

@app.route('/process_data', methods=['GET'])
def process_data():
    """
    Run data processing pipeline on unprocessed data
    """ 
    # Get all unprocessed samples from database
    # Get all unprocessed samples with full row data
    unprocessed_samples = db.session.query(Sample).filter(Sample.status == 'unprocessed').all()
    #sample = Sample.query.with_entities(*Sample.__table__.columns).all()
    
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

        ap_distance = 42

        # Update sample in database with ap_distance and mark as processed
        db.session.query(Sample).filter(Sample.id == samp["id"]).update({
            "ap_distance": ap_distance,
            "status": "processed"
        })
        db.session.commit()


    return 'processed data'





# # Configure Celery
# celery = Celery('tasks', broker='redis://localhost:6379/0')

# @celery.task
# def process_data_task():
#     """
#     Celery task to run data processing asynchronously
#     """
#     try:
#         process_data()
#         return {"status": "success"}
#     except Exception as e:
#         return {"status": "error", "message": str(e)}

# @app.route('/trigger_processing', methods=['POST'])
# def trigger_processing():
#     """
#     Endpoint to trigger asynchronous data processing
#     """
#     # Start the Celery task
#     task = process_data_task.delay()
    
#     return jsonify({
#         "message": "Data processing started",
#         "task_id": task.id
#     })

# @app.route('/task_status/<task_id>')
# def task_status(task_id):
#     """
#     Get the status of a processing task
#     """
#     task = process_data_task.AsyncResult(task_id)
#     if task.ready():
#         return jsonify({
#             "status": "completed",
#             "result": task.get()
#         })
#     return jsonify({
#         "status": "processing"
#     })


# @app.route('/most_recent_candidate', methods=['GET'])
# def most_recent_candidate():
#     """
#     Get the most recent candidate proposed by model
#     """
#     # load master csv file

#     # get candidates with status 'proposed'

#     # return proposed candidates, ranked by earliest to latest 

if __name__ == '__main__':
    app.run(debug=True) 
