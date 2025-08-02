import numpy as np
from tqdm import tqdm
import os

dir_path="/home/tester1/YT/Ben/SleepDiFFormer/data/datasets/GSS_datasets/ISRUC/seq"

seq_dir="/home/tester1/YT/Ben/SleepDiFFormer/data/datasets/GSS_datasets/ISRUC/seq2"

label_dir="/home/tester1/YT/Ben/SleepDiFFormer/data/datasets/GSS_datasets/ISRUC/labels2"


# print(f_names)
f_names = os.listdir(dir_path)
seq_f_names = []
label_f_names = []
print(f_names)
for f_name in f_names:
    if 'seq' in f_name:
        seq_f_names.append(f_name)
    if 'label' in f_name:
        label_f_names.append(f_name)

seq_f_names.sort()
label_f_names.sort()

print(seq_f_names)
print(label_f_names)

for seq_f_name in tqdm(seq_f_names):
   if not os.path.exists(label_dir + seq_f_name[:8]):
       os.makedirs(label_dir + seq_f_name[:8])

i = 0
for seq_f_name in tqdm(seq_f_names):
    print(seq_f_name)
    seqs = np.load(dir_path + seq_f_name)
    for seq in seqs:
        seq_name = f'/data/datasets/hang7_pre_100hz_seq/seq/{seq_f_name[:8]}/{seq_f_name[:8]}-{str(i)}.npy'
        with open(seq_name, 'wb') as f:
            np.save(f, seq)
        i += 1

j = 0
for label_f_name in tqdm(label_f_names):
    labels = np.load(dir_path + label_f_name)
    for label in labels:
        label_name = f'/data/datasets/hang7_pre_100hz_seq/labels/{label_f_name[:8]}/{label_f_name[:8]}-{str(j)}.npy'
        with open(label_name, 'wb') as f:
            np.save(f, label)
        j += 1

# import os
# import numpy as np
# from tqdm import tqdm

# # Example paths — adjust these if needed
# dir_path = '/home/tester1/YT/Ben/SleepDG/data/datasets/GSS_datasets/ISRUC'  # where your .npy input files are
# out_base = '/home/tester1/YT/Ben/SleepDG/data/datasets/hang7_pre_100hz_seq/'
# seq_dir = os.path.join(out_base, 'seq')
# label_dir = os.path.join(out_base, 'labels')

# # Collect file names
# f_names = os.listdir(dir_path)
# print(f_names)
# seq_f_names = sorted([f for f in f_names if 'seq' in f])
# label_f_names = sorted([f for f in f_names if 'label' in f])

# print("Seq files:", seq_f_names)
# print("Label files:", label_f_names)

# # Create necessary subfolders
# for f_name in seq_f_names + label_f_names:
#     group_name = f_name.split('_')[0]  # e.g., 'ISRUC-group1-1'
#     os.makedirs(os.path.join(seq_dir, group_name), exist_ok=True)
#     os.makedirs(os.path.join(label_dir, group_name), exist_ok=True)

# # Save sequences
# for seq_f_name in tqdm(seq_f_names, desc="Saving sequences"):
#     group_name = seq_f_name.split('_')[0]
#     seqs = np.load(os.path.join(dir_path, seq_f_name))
#     for i, seq in enumerate(seqs):
#         seq_path = os.path.join(seq_dir, group_name, f"{group_name}-{i}.npy")
#         with open(seq_path, 'wb') as f:
#             np.save(f, seq)

# # Save labels
# for label_f_name in tqdm(label_f_names, desc="Saving labels"):
#     group_name = label_f_name.split('_')[0]
#     labels = np.load(os.path.join(dir_path, label_f_name))
#     for j, label in enumerate(labels):
#         label_path = os.path.join(label_dir, group_name, f"{group_name}-{j}.npy")
#         with open(label_path, 'wb') as f:
#             np.save(f, label)
