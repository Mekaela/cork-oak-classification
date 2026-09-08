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