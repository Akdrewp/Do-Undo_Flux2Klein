import numpy as np
import os
from tqdm import tqdm

def strip_context(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = [f for f in os.listdir(input_dir) if f.endswith('.npz')]
    print(f"Filtering {len(files)} files...")

    for filename in tqdm(files):
        path = os.path.join(input_dir, filename)
        
        # Load the original data
        with np.load(path, allow_pickle=True) as data:
            # Create a dict of everything EXCEPT 'context'
            # We convert to a dict to ensure we actually pull the data into memory
            new_data = {key: data[key] for key in data.files if key != 'context'}
        
        # Save the compressed slim version
        save_path = os.path.join(output_dir, filename)
        np.savez_compressed(save_path, **new_data)

strip_context('processed_tuples', 'processed_tuples_slim')
strip_context('final_dataset', 'final_dataset_slim')