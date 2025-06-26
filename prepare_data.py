import os
import multiprocessing as mp
from configs.config import data_cfg_dict

print("Command List before: ", arg_parts)  # Check the commands being added
arg_parts = [
    'python utils/raw_data.py',
    'python utils/raw_semi_data.py'
]
print("Command List after: ", arg_parts)  # Check the commands being added
pool = mp.Pool(processes=6)
pool.map(os.system, arg_parts)
pool.close()
pool.join()

arg_parts = []
print("Starting to check data configurations...")  # This should appear in the output
for v in data_cfg_dict:
    print("Data Config Dictionary: ", data_cfg_dict)  # This will show the contents of the dictionary
    # Print the current configuration and the augmentation flag
    use_augment = data_cfg_dict[v].USE_AUGMENT
    print(f"Checking config for version {v}: USE_AUGMENT = {use_augment}")
    
    if use_augment:
        arg_parts.append(f'python utils/aug_data.py --version {v}')
        print(f"Augmentation is enabled, running 'aug_data.py' for version {v}")
    else:
        arg_parts.append(f'python utils/data.py --version {v}')
        print(f"Augmentation is not enabled, running 'data.py' for version {v}")
pool = mp.Pool(processes=6)
pool.map(os.system, arg_parts)
pool.close()
pool.join()
