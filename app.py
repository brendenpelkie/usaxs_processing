from flask import Flask, render_template, request, jsonify
import os
from flask_sqlalchemy import SQLAlchemy
from celery import Celery

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
    id = db.Column(db.Uuid(), primary_key=True)
    scattering_fp = db.Column(db.String(100), nullable=True)
    teos_vf = db.Column(db.Float(), unique=False, nullable=False)


    def __repr__(self):
        return f'<User {self.name}>'

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
        sample = Sample(
            id=data['uuid'],
            scattering_fp=file_path,
            teos_vf=data['teos_vf']
        )

        db.session.add(sample)
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
    sample = Sample.query.all()
    

    print(sample)
    return jsonify(sample)

# 
# def process_data():
#    """
#    Run data processing pipeline on unprocessed data
#    """
#    pass


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
