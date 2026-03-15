import os
import json
import numpy as np
from decord import VideoReader, cpu
from tqdm import tqdm

# --- Configuration ---
OBJ_MAP = {
  "C1": "Toy Car", "C2": "Mug", "C3": "Laptop", "C4": "Storage Furniture",
  "C5": "Bottle", "C6": "Safe", "C7": "Bowl", "C8": "Bucket", 
  "C9": "Scissors", "C11": "Pliers", "C12": "Kettle", "C13": "Knife",
  "C14": "Trash Can", "C17": "Lamp", "C18": "Stapler", "C20": "Chair"
}

DO_VERBS = {'pickup', 'open', 'push', 'pour', 'dump', 'fill', 'cut', 'clamp', 'turn', 'lift'}
UNDO_VERBS = {'putdown', 'close', 'pull', 'stop', 'release', 'leave'}

root_anno = "./datasets/HOI4D_annotations"
root_color = "./datasets/HOI4D_release"
output_dir = "./processed_tuples"
os.makedirs(output_dir, exist_ok=True)

def extract_7_frames_decord(video_path, t_start, t_end, margin=0.10):
  """ 
  Extracts 7 frames dynamically squeezed inside the action duration.
  margin=0.15 means we skip the first 15% and last 15% of the action.
  """
  vr = VideoReader(video_path, ctx=cpu(0), width=512, height=512)
  fps = vr.get_avg_fps()
  
  # Calculate the total duration of the action in seconds
  duration = t_end - t_start
  
  # Squeeze the start and end times inward by the margin percentage
  safe_start_t = t_start + (duration * margin)
  safe_end_t = t_end - (duration * margin)
  
  # Convert to actual frame indices
  start_f = int(safe_start_t * fps)
  end_f = min(int(safe_end_t * fps), len(vr) - 1)
  
  # Failsafe: If the action is incredibly short, just use the raw frames
  if start_f >= end_f:
      start_f = int(t_start * fps)
      end_f = min(int(t_end * fps), len(vr) - 1)
  
  # Sample 7 frames evenly across this new, safer "meat" of the action
  indices = np.linspace(start_f, end_f, 7, dtype=int)
  frames = vr.get_batch(indices).asnumpy()
  
  return frames


"""
The dataset consists of videos of actions and annotated desriptions and timestamps
of each action.

For Do-Undo seperate all verbs that have a do and an undo action related to them
from
https://github.com/leolyliu/HOI4D-Instructions/blob/main/definitions/task/task_definitions.csv

For each video go to the annotation and find if there is some Do verb that is eventually
followed by an Undo verb to add frames to the dataset

1. Get annotated files
For file in annotated files:
  2. Parse events to find Do Undo pairs
  3. Get corresponding video path
  4. Get 7 temporal frames
  5. Save to disk


- Io: Start Do frame
- If: Final Undo frame
- context: 5 intermediate  Io and If.
- vb: array containing the [verb_do, verb_undo]
- obj: Object being acted upon

"""

# 1. Search for all files
all_files = []
for root, _, files in os.walk(root_anno):
  if "color.json" in files:
    all_files.append((root, os.path.join(root, "color.json")))

dataset_count = 0

# tqdm for progress
for root, json_file in tqdm(all_files, desc="Harvesting HOI4D", unit="video"):
  parts = root.split(os.sep)
  cat_id = next((p for p in parts if p.startswith('C') and p[1:].isdigit()), None)
  
  if cat_id is None:
    continue
  
  obj_name = OBJ_MAP.get(cat_id, "Object")

  with open(json_file, 'r') as f:
    try:
      data = json.load(f)
      events = data['events']
    except: # Events is empty some times probably oversight by team
      continue 

    for i, e in enumerate(events):
      verb_do = e['event'].lower()
      
      # 2. Get Do-Undo pair
      if verb_do in DO_VERBS:
        for next_e in events[i+1:]:
          verb_undo = next_e['event'].lower()
          
          if verb_undo in UNDO_VERBS:

            # 3. Get video path
            t_start = e['startTime']
            t_end = next_e['endTime']
            
            rel_path = os.path.relpath(root, root_anno)
            clean_rel_path = rel_path.replace("/action", "").replace("\\action", "")
            video_path = os.path.join(root_color, clean_rel_path, "align_rgb/image.mp4")
            
            if os.path.exists(video_path):
              # 4. Extract frames
              frames = extract_7_frames_decord(video_path, t_start, t_end)
              
              #5. Save the frames, object, and action
              if frames is not None and len(frames) == 7:
                dataset_count += 1
                np.savez_compressed(
                  os.path.join(output_dir, f"sample_{dataset_count:05d}.npz"),
                  Io=frames[0],
                  If=frames[-1],
                  context=frames[1:6],
                  vb=np.array([verb_do, verb_undo]),
                  obj=obj_name
                )
            break 

print(f"\n✅ Created {dataset_count} compressed tuples in '{output_dir}'")