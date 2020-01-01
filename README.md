# GreekLyricsClassification
Text Mining and Natural Language Processing | MSc Data &amp; Web Science | AUTh
---
## Contributors
@Dimitris Papadopoulos @Dimitris Apostolou
---

This project aims to build a system that can identify the genre of a greek song based on its lyrics. 

---


### To run the code, follow the steps below.

To import the environment we are working on, open a terminal at /resources, where the environment.yml is located and run:
*conda env create -f environment.yml*
If an error shows up about the installation of spacy library, we will fix it below.

To activate the imported environment: 
*conda activate python3.7*

Then we need to install spacy to the activated environment, as we use it for lemmatization. Unfortunately the greek model is not yet upoloaded into spacy, but it is upoloaded in spacy nightly which is not supported by anaconda.

To install spacy nightly:
*pip install spacy-nightly*
*conda install -c conda-forge spacy-lookups-data*

And finally to install the greek model:
*python -m spacy download el_core_news_sm*

The code is placed in jupyter's notebook file, Classifier.ipynb .

### Loading of pre-trained Neural Network models

While you run the code you can either train the neural models your self, or load them from /resources, where we uploaded them as soon as we trained them. To choose to train the models just comment the model loading shell and uncomment the model trainning shell above.


 
