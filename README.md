# Assignment 2

- Hritvik Choudhari 
- 119208793

## Running the code

Modify the root_location variable in dataset_location.py to point to the location of the dataset.

To run the code check the command

bash

python fit_data.py --type "vox"

python fit_data.py --type "point" --lr 4e-2

python fit_data.py --type "mesh"

python train_model.py --type "vox"

python eval_model.py --type "vox" --load_checkpoint

python train_model.py --type "point" --lr 4e-5 --batch_size 4 --num_workers 4

python eval_model.py --type "point" --load_checkpoint --batch_size 4 --num_workers 4

python train_model.py --type "mesh" --lr 4e-5 --batch_size 4 --w_smooth 2 --num_workers 4

python eval_model.py --type "mesh" --load_checkpoint --batch_size 4 --w_smooth 2 --num_workers 4



## Results

Results are stored in the results folder for Task 1. For Task 2, the results are stored in the results folder with each model having its own folder.