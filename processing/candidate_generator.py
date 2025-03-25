import uuid
import numpy as np
from models.sample import create_sample, db

def generate_candidate():
    """
    Generate a new candidate sample with random parameters
    
    Returns:
        dict: Information about the generated candidate
    """
    # Generate random parameters
    data = {
        'uuid': str(uuid.uuid4()),
        'teos_vf': np.random.random(),
        'ammonia_vf': np.random.random(),
        'ethanol_vf': np.random.random(),
        'water_vf': np.random.random(),
        'ctab_mass': np.random.random(),
        'f127_mass': np.random.random()
    }

    # Create new sample with proposed parameters
    sample = create_sample(data)
    sample.status = 'proposed'
    db.session.commit()

    return {
        "message": "New candidate proposed",
        "uuid": data['uuid'],
        "teos_vf": data['teos_vf']
    } 