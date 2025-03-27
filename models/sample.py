from flask_sqlalchemy import SQLAlchemy
from flask import current_app

db = SQLAlchemy()

class Sample(db.Model):
    id = db.Column(db.String(36), primary_key=True)
    scattering_fp = db.Column(db.String(100), nullable=True) 
    background_fp = db.Column(db.String(100), nullable=True)
    teos_vf = db.Column(db.Float(), unique=False, nullable=False)
    ammonia_vf = db.Column(db.Float(), unique=False, nullable=False)
    ethanol_vf = db.Column(db.Float(), unique=False, nullable=False)
    water_vf = db.Column(db.Float(), unique=False, nullable=False)
    ctab_mass = db.Column(db.Float(), unique=False, nullable=False)
    f127_mass = db.Column(db.Float(), unique=False, nullable=False)
    sample_order = db.Column(db.Integer, unique=True, nullable=False)
    status = db.Column(db.String(20), default='unprocessed')
    amplitude_distance = db.Column(db.Float(), unique=False, nullable=True, default=None)
    phase_distance = db.Column(db.Float(), unique=False, nullable=True, default=None)
    hiq_peak = db.Column(db.Float(), unique=False, nullable=True, default=None)

    def __repr__(self):
        return f'{self.id}, {type(self.id)}'

def create_sample(data, scattering_fp=None, background_fp=None):
    """
    Create a new sample in the database with auto-incrementing sample_order
    
    Args:
        data (dict): Dictionary containing sample data
        scattering_fp (str, optional): Path to the scattering data file
        background_fp (str, optional): Path to the background data file
        
    Returns:
        Sample: The newly created sample object
    """
    # Get the maximum sample_order value
    max_order = db.session.query(db.func.max(Sample.sample_order)).scalar()
    next_order = 1 if max_order is None else max_order + 1

    # Create a new sample
    sample = Sample(
        id=data['uuid'],
        teos_vf=data.get('teos_vf'),
        ammonia_vf=data['ammonia_vf'], 
        ethanol_vf=data['ethanol_vf'],
        water_vf=data['water_vf'],
        ctab_mass=data['ctab_mass'],
        f127_mass=data['f127_mass'],
        sample_order=next_order
    )

    # Assign any additional entries from data dict to sample
    for key, value in data.items():
        if hasattr(sample, key) and key not in ['uuid', 'teos_vf', 'ammonia_vf', 'ethanol_vf', 'water_vf', 'ctab_mass', 'f127_mass']:
            setattr(sample, key, value)
    
    # Set file paths if provided
    if scattering_fp:
        sample.scattering_fp = scattering_fp
    if background_fp:
        sample.background_fp = background_fp
    
    db.session.add(sample)
    db.session.commit()
    return sample 