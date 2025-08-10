import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.cm as cm
from pathlib import Path
import gc
import re
global_token_index = 30
patch_tokens = 30
signal_len = 3000
patch_size = signal_len // patch_tokens
cmap = cm.get_cmap('viridis')
label_names = ["Wake", "N1", "N2", "N3", "REM"]
dataset_names = ['sleep-edfx', 'HMC', 'ISRUC', 'SHHS1', 'P2018']
# Cross-domain cast
cross_domain_map = {
    "HMC": ['sleep-edfx', 'ISRUC', 'SHHS1', 'P2018'],
    "ISRUC": ['sleep-edfx', 'HMC', 'SHHS1', 'P2018'],
    "P2018": ['sleep-edfx', 'HMC', 'ISRUC', 'SHHS1'],
    "SHHS1": ['sleep-edfx', 'HMC', 'ISRUC', 'P2018'],
    "sleep-edfx": ['HMC', 'ISRUC', 'SHHS1', 'P2018']
}
num_sample = [4,10,15,15,0]#manualy selet the sample index for each class
fig, axes = plt.subplots(5, 2, figsize=(20,8), gridspec_kw={'height_ratios': [1, 1, 1, 1,1],'width_ratios':[1,1]})
fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.05,
                    wspace=0.05, hspace=0.03)
for ch in range(2):
    for num in range(0+ch*5,5+ch*5):
        npz_path = Path(rf"C:\Users\saved_sequences_sample\class_{num%5}.npz") #change the path to user's own
        sample_idx = num_sample[num%5]
        # ==== auto-get target_name ====
        target_name = None
        for part in npz_path.parts:
            if part.startswith("saved_sequences_"):
                target_name = part.replace("saved_sequences_", "")
                break
        if target_name is None or target_name not in cross_domain_map:
            raise ValueError(f"the legal target domain name is not included in the path（got: {target_name}）")
        #get dataset index
        match_dataset = re.search(r'dataset_(\d+)', str(npz_path))
        if not match_dataset:
            raise ValueError("can not find dataset index")
        dataset_index = int(match_dataset.group(1))
        #get class index
        match_class = re.search(r'class_(\d+)', npz_path.name)
        if not match_class:
            raise ValueError("can not find class index")
        class_index = int(match_class.group(1))
        dataset_real_name = cross_domain_map[target_name][dataset_index]
        label_name = label_names[class_index]
        data = np.load(npz_path)
        signal = data['signal']
        attn = data['attn']
        class_output_dir = npz_path.parent
        class_output_dir.mkdir(exist_ok=True)
        all_attn_values = [
            attn[layer_idx, sample_idx][global_token_index, :patch_tokens]
            for layer_idx in range(4)]
        attn_min = min([v.min() for v in all_attn_values])
        attn_max = max([v.max() for v in all_attn_values])
        norm = plt.Normalize(vmin=attn_min, vmax=attn_max)
        sm = cm.ScalarMappable(cmap=cmap,norm=norm)
        channels = ['EEG', 'EOG']
        eeg = signal[sample_idx]
        ax = axes.T.flatten()[num]
        ax.set_xlim(0, signal_len)
        ax.tick_params(axis='y', labelsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.margins(y=0)
        ax.plot(eeg[ch], color='black', linewidth=0.8, label=channels[ch])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x / 100)}s'))
        ax.yaxis.tick_right()
        ax.set_ylim(-4.2, 4.0)
        ax.set_yticks(np.linspace(-4.0, 4.0, 5))
        ax.tick_params(axis='y', direction='in', pad=5)
        if num!=4 and num!=9: #do not display x-axis coordinate value
            ax.set_xticklabels([])
        if num!=5:            #do not display y-axis coordinate value
            ax.set_yticklabels([])
        for layer_idx in range(4):
            attn_map = attn[layer_idx, sample_idx]
            global_attn = attn_map[global_token_index, :patch_tokens]
            norm_attn = (global_attn - global_attn.min()) / (global_attn.max() - global_attn.min() + 1e-8)
            for i in range(patch_tokens):
                start = i * patch_size
                end = start + patch_size if i < patch_tokens - 1 else signal_len
                a=norm(global_attn[i])
                color = cmap(a)
                ymin, ymax = ax.get_ylim()
                height = (ymax-ymin) / 4
                rect = patches.Rectangle(
                    (start, ymax - (layer_idx + 1) * height), patch_size, height,
                    linewidth=0, facecolor=color, alpha=0.6)
                ax.add_patch(rect)
            #add the horizontal lines
            for l in range(1, 4):
                y_pos = ymax - l * height
                ax.axhline(y=y_pos, color='gray', linestyle='--', linewidth=0.5, alpha=0.6)
            if num==6: #display layer marks
                ax.text(signal_len+10, ymax- (layer_idx + 1) * height+height/2, f"Layer{layer_idx+1}", fontsize=14, ha='left', va='center')
        if num<5:
            ax.set_ylabel(f"{label_names[num%5]}",fontsize=15)
plt.tight_layout(rect=(0, 0, 0.98,0.95))
fig.text(0.054, 0.92, "EEG", ha='center', va='bottom', fontsize=15)
fig.text(0.54, 0.92, "EOG", ha='center', va='bottom', fontsize=15)
cbar_ax = fig.add_axes((0.955, 0.04, 0.005, 0.53))
fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
cbar_ax.tick_params(axis='y', labelsize=8)
cbar_ax.yaxis.set_label_position('left')
cbar_ax.set_ylabel("Attention Weight",fontsize=10)
save_path = class_output_dir / f"4layer_ofEEG&EOG_visualize.png"
plt.savefig(save_path, dpi=1000,bbox_inches='tight')
plt.show()
plt.close(fig)
gc.collect()
print(f"saved: {save_path.name}")

