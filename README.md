# AutoDL-DDoS-Defense
Efficient AutoDL for Generating Denial-of-Service Defense Models in the Internet of Things

## Project Description
Introduce the project.

### Folder Description:
* data: Contains scripts or notebooks related to data preparation and data extraction. The original CICDDoS2019 data should be placed in the data/origin_data/ directory.
    * data.sh: Script to execute the code in the data folder. The execution order is as follows: 1. extract.py, 2. cleaning.py, 3. divide.py.
    * extract.py: Randomly selects subdata, including five percent of DDoS packets and all benign packets.
    * cleaning.py: Cleans the data.
    * divide.py: Divides the clean data into the feature and the label CSV files.
* data/predict_data: Contains scripts or notebooks related to predicting data preparation. This data is used for testing the prediction time of the ML/DL pipeline.
* main: Contains scripts or notebooks related to searching for the best configuration and training (refit) with it.
    
    Contain our nine test cases described below:
    1. DC: Default configuration.
    2. DCwM: Default configuration with MinMaxScalar.
    3. DCwR: Default configuration with RobustScalar.
    4. DCwP: Default configuration with PCA.
    5. DCwF: Default configuration with FeatureAgglomeration.
    6. LW1: Lightweight architecture LNN.
    7. T1: The architecture similar to LW1 while without the lightweight mechanism.
    8. LW2: Lightweight architecture two, CLNN.
    9. T2: The architecture similar to LW2 while without the lightweight mechanism.
* search_space: Contains our lightweight model search space.

## Installation
### Dependencies:
* Python >= 3.7
* autopytorch == 0.2.1

### Installation step:

## Usage

## License