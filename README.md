# cork-oak-classification
For thesis cork oak height and biomass classifications


Install notes:
Make sure python is installed. 
Create new virtual environment:
python -m venv .venv
activate venv (or similar based on your OS):
.venv\Scripts\activate.bat

note: some of this may be avoidable if using conda rather than venv. Some other errors may occur if other tools are not updated on computer (such as C++ Build Tools)
Install wheel and GDAL:
pip install --upgrade pip setuptools wheel
pip install numpy

With conda instead. Download Anaconda
- make sure path includes (for windows, still needed on mac, but is different):
C:\Users\<YourUsername>\Anaconda3
C:\Users\<YourUsername>\Anaconda3\Scripts
C:\Users\<YourUsername>\Anaconda3\Library\bin

create conda environment:
conda create -n gdal_env python=3.13
activate it: conda activate gdal_env
install GDAL: conda install -c conda-forge -y gdal fiona rasterio geopandas pyproj netcdf4

install requirements.txt without using pip for these libraries:
pip install --no-binary gdal,fiona,rasterio,geopandas,pyproj -r requirements.txt

you will need to have the ipykernel installed. your vscode should alert you and ask if you´d like it installed

For installing detectree2:
install detectron2:
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
install detectree2: (or do the first step before running requirements.txt, though might not work as it needs torch,torchvision,torchaudio)
python -m pip install detectree2

Also need to install setuptools (must be old version):
conda install "setuptools<81"

note: to get GPU working, the pytorch must be a cuda version suited to the user´s CUDA version

## Data preparation



## Training models
YOLO: navigate to src/models/training
- ensure that data.yaml points to the correct train and val folders created in data preparation
- run train_yolov8.py

Detectree2: navigate to src/models/training/train_detectron2.py'
for multiple images/plots:
- modify the 'train_sites' and 'val_sites' to your sites
- uncomment create_tiles_across_sites() in main()
- comment out create_training_tiles() in main()
- run train_detectron2.py

for one image/site:
- modify img_path and crown_path in main() to your images
- run train_detectron2.py


## Predicting
YOLO: navigate to src/models/predicting
for small image (the size of plots in this study ~ 30m diameter)
- make sure images to predict follow the path convention in the for loop in main()
- run predict_yolo.py

Detectree2:
- change TEST_FILE to the file with the test tiles. Use the tile method if not already tiled. 
- move the resulting predictions folder (generated in the root folder) to PRED_PATH - this keeps the predictions out of the root folder
- comment predict() out and uncommment project() and stitch(). run



## Validation metrics
The files val_main calls all of the validation metrics for both detection and classification, as well as height and crown area metrics. 
- The reference path 'data/plots' must contain the 3 band RGB, and the CHM of each plot
- The model must be edited to be the model the user wants the validation metrics for, on line 8
- The predictions must be in the relevant prediction directory, as on line 11
- The reference csv must be specified on line 17, with the columns: location_id,plot_id,tree_id,ht_m
- The crown area csv must be specified on line 18, with the columns: tree_id,area_m2


## pipeline to predict crowns and estmate heights and crowns
main.py calls functions from other files to prepare required files and predict crowns, then estimate individual heights and crown areas. 
To use this
- make sure the RGB and DSM variables point to the images you want to predict
- change model to either YOLO or Detectree2, depending on which model you'd like to use to detect the crowns.
- run the file

Outputs: 
- prints in the terminal - tree count, total crown area
- outputs csv of individual tree heights and crown areas
