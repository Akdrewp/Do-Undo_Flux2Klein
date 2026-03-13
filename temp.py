import os
import shutil
import random
from tqdm import tqdm

def split_test_set(npz_src, json_src, npz_test, json_test, num_samples=300):
    """Separates 300 random samples from training folders into test folders.
    
    1. Identifies common sample IDs between images and metadata.
    2. Randomly selects the specified number of IDs.
    3. Moves the files to the new destination directories.
    """
    # Create test directories if they don't exist
    os.makedirs(npz_test, exist_ok=True)
    os.makedirs(json_test, exist_ok=True)

    # Get matching IDs (stripping extensions and suffixes)
    npz_files = {f.replace(".npz", "") for f in os.listdir(npz_src) if f.endswith(".npz")}
    json_files = {f.replace("_meta.json", "") for f in os.listdir(json_src) if f.endswith(".json")}
    
    common_ids = list(npz_files.intersection(json_files))
    
    if len(common_ids) < num_samples:
        raise ValueError(f"Not enough samples! Found {len(common_ids)}, needed {num_samples}")

    # Randomly pick 300 IDs
    test_ids = random.sample(common_ids, num_samples)
    print(f"📦 Moving {num_samples} samples to test folders...")

    for sample_id in tqdm(test_ids):
        # Define source and destination paths
        npz_name = f"{sample_id}.npz"
        json_name = f"{sample_id}_meta.json"
        
        # Move NPZ
        shutil.move(os.path.join(npz_src, npz_name), os.path.join(npz_test, npz_name))
        
        # Move JSON
        shutil.move(os.path.join(json_src, json_name), os.path.join(json_test, json_name))

    print(f"✅ Split complete! {num_samples} samples moved to {npz_test} and {json_test}")

# Execute the split
split_test_set(
    npz_src='processed_tuples_slim', 
    json_src='final_dataset', 
    npz_test='test_tuples_slim', 
    json_test='test_final_dataset', 
    num_samples=300
)