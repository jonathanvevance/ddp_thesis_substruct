<div id="top"></div>

<!-- PROJECT LOGO -->
<br />
<div align="center">

<h2 align="center">Reaction Outcome Prediction (Thesis)</h2>

  <p align="center">
    Using MACCS Keys Substructures and Graph Neural Networks
    <br />
    <a href="#about-the-project"><strong>Explore the docs »</strong></a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->

## About The Project

This is a DL based approach devised for reaction outcome prediction as part of my IITM MTech thesis. It uses the chemical substructures from MACCS fingerprinting to simplify the reaction, and solves the simplified reaction. The first step is a classification problem, where the reactivity scores of all substructure pairs are predicted. The second step is a regression problem of predicting the adjacency matrix (bonds) in the RHS of the simplified reactions. Finally we combine these predictions using a simple strategy. 

<p align="right">(<a href="#top">back to top</a>)</p>

### Built With

- [Pytorch](https://pytorch.org//)
- [Pytorch-Geometric](https://pytorch-geometric.readthedocs.io/en/)
- [RDKit-Python](https://www.rdkit.org/docs/GettingStartedInPython.html)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- GETTING STARTED -->

## Getting Started
Note that the code has been run and verified on Ubuntu as well as Windows systems. The instructions that follow are confirmed to be working on an Ubuntu system. 

### Setting up the conda environment

1. Create a new Anaconda environment
   ```sh
   conda create --name envname python=3.7
   ```

2. Install Pytorch 1.11 (1.11 is the latest version as of writing)
   ```sh
   conda install pytorch cudatoolkit=11.3 -c pytorch
   ```

3. Install Pytorch-Geometric suitable for Pytorch 1.11
   ```sh
   pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
   pip install torch-geometric
   ```

4. Install latest RDKit from conda-forge repository (2022.03.2 at the time of writing)
   ```sh
   conda install -c conda-forge rdkit
   ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

### Preparing the dataset for training

1. Place the training dataset (text file with reaction smiles) in data/raw as train.txt.
2. Activate the conda environment. 
4. From the root directory, run
   ```sh
   python src/prepare_dataset.py
   ```

### Training the classification model:

1.  After <a href="### Preparing the dataset for training">preparing the dataset</a>:

        eval_dataset
            ├── scoring_clips
                ├── clip_1.mp4
                ├── clip_2.mp4
                ├── ...
            ├── non_scoring_clips
                ├── clip_4.mp4
                ├── clip_6.mp4
                ├── ...

2.  2. From the root directory, activate the python venv by running:
    ```sh
    source .env/bin/activate
    ```
3.  Edit the src/configs/eval_config.py file with the required evaluation settings.
4.  From the root directory, start training by running:
    ```sh
    python src/evaluate.py
    ```

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ROADMAP -->

## Roadmap

- [] Interactive video player with seek bar in the web application.
- [] Inference speedup using TensorRT on GPU and Intel OpenVino on Intel CPU.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- CONTACT -->

## Contact Us

<!-- Jonathan Ve Vance - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com -->

1. Jonathan Ve Vance - [Linkedin](https://linkedin.com/in/jonathanvevance) - jonathanvevance@gmail.com
2. Irfan Thayyil - [LinkedIn](https://www.linkedin.com/in/mohammed-irfan-thayyil-34311a166) -irfanthayyil@gmail.com
3. Adil Muhammed K - [LinkedIn](https://www.linkedin.com/in/adil-mohammed-065603155) - adilmohammed2000@outlook.com
4. Akshay Krishna - [LinkedIn](https://www.linkedin.com/in/akshaykrishh/) - akshaykrishnakanth@gmail.com

Project Link: [https://github.com/jonathanvevance/basketall_scoring_detection](https://github.com/jonathanvevance/basketall_scoring_detection)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

- []() We obtained pretrained weights for basket (hoop) detector yolov3 model from <a href = "https://github.com/SkalskiP/ILearnDeepLearning.py"> this great repository</a>. Huge shoutout to the author.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

[product-screenshot]: readme_images/app_screenshot.png

ddp_thesis_substruct
==============================

Reaction outcome prediction (IITM DDP Thesis)

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
