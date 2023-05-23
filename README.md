# AutoDL-DDoS-Defense
Efficient AutoDL for Generating Denial-of-Service Defense Models in the Internet of Things

## Project Description
Efficient AutoDL for Generating Denial-of-Service Defense Models in the Internet of Things

### Folder Description:
* autopytorch_0.2.1: Source code of [autopytorch](https://github.com/automl/Auto-PyTorch) with our lightweight search space integration.

* data: Contains scripts and notebooks related to data preparation and extraction.
    * evaluation_data: Put the code to generate evaluation data, we only use testing day data to evaluate.
    * predict_data: Scripts for preparing data used to test the prediction time of the ML/DL pipeline.
    * searching_data: Scripts for the search and refit phases.
    * origin_data: The original CICDDoS2019 dataset. Please provide the data in the required format: origin_data/CSV-01-12/01-12/ ...

* main: Contains scripts or notebooks related to searching for the best configuration and training (refit) with it.
    * search and refit folder:
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
    * evaluation: Put the code to evaluate the model generated in the refit phase.
    * predict: Put the code to get the predict time of the model generated in the refit phase.

* search_space: Contains our lightweight model search space.

## Installation
### Dependencies:
* Python >= 3.7
* autopytorch == 0.2.1

### Installation step:
1. `pip install autopytorch==0.2.1`, we use this command mainly to get the dependencies of autopytorch, you should use our autopytorch source code with our lightweight search space integration.

## Usage
1. Run all the scripts in the data folder to prepare data.
2. Run the main/search/search.sh to search for the best ML/DL pipeline configuration
3. Parse the best configuration string to the code in the refit folder (a string).
4. Run the main/refit/refit.sh to refit the model.
5. Run main/predict/predict.sh and main/evaluation/evaluation.sh to get the predict time and evaluation result of different models.

## License
MIT License
