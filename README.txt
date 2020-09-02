### Command to Run (SANet module replaced by MetaAdaIN) ###
python train_merged.py --content_dir 'Enter_Path_Here' --style_dir 'Enter_Path_Here' --save_model_interval 10000 --max_iter 80000

## If model is partially trained and need to resume training from the last checkpoint, then add the --start_iter flag ###
python train_merged.py --start_iter 'enter_last_checkpoint_no(ex: 10000,20000 etc)' --content_dir 'Enter_Path_Here' --style_dir 'Enter_Path_Here' --save_model_interval 10000 --max_iter 80000

### Command (Original SANet) ###
python train.py --content_dir 'Enter_Path_Here' --style_dir 'Enter_Path_Here' --save_model_interval 10000 --max_iter 80000

