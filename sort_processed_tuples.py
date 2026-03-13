import os
import shutil

source_dir = "processed_tuples"
dest_dir = "static_tuples_removed"
txt_file = "static_filenames.txt"

os.makedirs(dest_dir, exist_ok=True)

with open(txt_file, 'r') as f:
    filenames = [line.strip() for line in f if line.strip()]

moved_count = 0
missing_count = 0

for fname in filenames:
    src_path = os.path.join(source_dir, fname)
    dst_path = os.path.join(dest_dir, fname)
    
    if os.path.exists(src_path):
        shutil.move(src_path, dst_path)
        moved_count += 1
    else:
        missing_count += 1

print(f"Successfully moved {moved_count} static files.")
if missing_count > 0:
    print(f"Skipped {missing_count} files that were already moved or not found.")