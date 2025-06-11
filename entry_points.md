Frist of all, set data paths in "SETTINGS.json".  

Training steps:    
1. Run 'python prepare_data.py'.   
When it's finised, you can find all prepared data under the path "data_cache_dir" ("data_cache/" by default) set in "SETTINGS.json".   

2. Run 'python train.py'.   
When it's finised, you can find all trained models under the setting "model_dir" ("models/" by default) specified in "SETTINGS.json".  
   
Inference steps:   
Run 'python predict.py'.   
When it's finised, you can find the submission csv file under the setting "submission_dir" ("output/" by default) specified in "SETTINGS.json".  
