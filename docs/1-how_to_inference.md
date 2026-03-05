## 🧪 How to Run Inference on Your Own Data

This guide explains how to apply the NeuroLex pipeline to your own dataset.

---

### **Step 1. Prepare your data**

Organize your data into a folder where **each subject is stored as a `.pkl` file**.

Each file should be a Python dictionary with the following keys:

* `timeseries`: fMRI time series (NumPy array)
* `age`: subject age
* `sex`: subject sex (`male` / `female`)
* `site`: site identifier (can be `"anonymous"`)

#### Example

```python
import os
import numpy as np
import pickle

save_dir = 'your_path/train'
os.makedirs(save_dir, exist_ok=True)

tmp = dict()
dat = np.load(file_path)  # path/to/your/timeseries.npy

tmp['timeseries'] = dat
tmp['age'] = row['age']
tmp['sex'] = row['sex']
tmp['site'] = 'anonymous'

save_path = os.path.join(save_dir, f"{name}.pkl")  # define your own naming
with open(save_path, 'wb') as f:
    pickle.dump(tmp, f)
```

---

### **Step 2. Infer Brain State Tokens (BSTs)**

Run token inference using the pretrained NeuroLex model:

```bash
python 01-infer_tokens.py \
    --model_path model/model.pth \
    --input_data your_path/train \
    --output_path results/token_results/train
```

> 💡 You can organize different datasets into separate folders (e.g., `train/`, `test/`, `clinical/`).

---

### **Step 3. Compute Temporal Features**

Extract temporal dynamics from token sequences:

```bash
python 02-gather_features.py \
    --token_folder results/token_results \
    --save_file results/temporal_features.csv
```

This step computes:

* **Fractional Occupancy (FO)**
* **Dwell Time (DT)**
