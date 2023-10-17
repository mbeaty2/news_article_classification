# Developing a Classification System for News Articles using an LSTM

This project implements an LSTM classification system to classify news articles as Real or Fake. 

## Description

Using an open-source Kaggle dataset containing news articles and their labelled classifications as "Real" or "Fake", I developed an LSTM model capable of accurately predicting a test article's classification. This project was developed in response to the onslaught of false media releases occuring during the 2016 and 2020 U.S. presidential campaigns. At a time where misinformation was rampant, public opinion was often swayed towards believing in false narratives. This project seeks to pose an alternative method and asks if it is possible for an AI to determine for me (the reader) whether the information I am about to consume is real or fake. If so, can I reduce my intake of misinformation and false narratives. 

This project was developed in part of the Natural Language Processing course completed during the spring term of my Masters of Data Science and AI. To fit the hand-in specifications, the project was originally written and tested within a jupyter notebook that has been uploaded to this repository. This notebook has also been adapted to python files to run in the command line. These include a file containing all functions and a file containing all command line arguments. The additional files within this repository hold the saved model and weights from my trained classification system should you wish to build off the results of this work rather than re-train a model yourself. Additionally, a PDF file containing project report and insights into the research conducted can also be found within this repository. 

## Getting Started

### Dependencies

* The foundation of this project is within tensorflow. A keras compatible engine is required.
* This project utilized a 16GB GPU. 

### Installing

* Access and download the Kaggle dataset here: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
* NLTK "stopwords" corpus available at: https://www.nltk.org/search.html?q=stopwords
* GoogleNews Vectors available at: https://www.kaggle.com/datasets/leadbest/googlenewsvectorsnegative300
* The rest of the files can be adapted using either the Jupyter Notebook above, or the python files.

### Executing program

* Download the necessary datasets. 

If using the Jupyter Notebook:
* Open the file in a code editor and connect to a code environment.
* Run each cell in the notebook. Should you want to make adjustments, be sure to save the file before continuing to run each cell. If there are issues, restart the kernel and re-run each cell.

If using the python files:
* Download the .py files
* Open a terminal and locate where the downloaded datasets and files are on your local machine.
* Create a new environment to run everything in.
* Run this line in your terminal: python main.py -d <data_directory> --clean-data --topic-modeling --save-model

## Help

The most common problem I encountered was run-time lengths and errors. I encourage those interested in pursuing this project to ensure they have a strong enough GPU to handle the large datasets. Should trouble still occur, try reducing the epoch size, the training set size, the batch size, or running the notebook in Google Colab.  

## Authors

Marissa Beaty, https://github.com/mbeaty2

## Version History

* 0.2
    * translation of the project from a jupyter notebook to a python file for mass use and adaptation.
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
