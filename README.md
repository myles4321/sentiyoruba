# Yoruba Movie Sentiment Analysis
## Introduction
This project aims to perform sentiment analysis on Yoruba movie reviews. The goal is to analyze the sentiment of the reviews and classify them as positive, negative, or neutral. The sentiment analysis will help gain insights into the audience's perception and opinions about Yoruba movies.

## Dataset
The dataset used for this project consists of Yoruba movie reviews collected from various online platforms, it is a yosm: A new yoruba sentiment corpus for movie reviews. It includes a collection of 1500 reviews written in Yoruba language, along with their corresponding sentiment labels (positive, negative, or neutral). The dataset is manually labeled by human annotators to ensure accuracy. find link to dataset here: https://lanfrica.com/record/yosm-a-new-yor-b-sentiment-corpus-for-movie-reviews

## Technology Stack
The project is developed using the following technologies and tools:

* Python: The programming language used for implementing the sentiment analysis algorithms and data processing tasks.
* Natural Language Processing (NLP) Libraries: NLP libraries such as NLTK are utilized for text preprocessing, feature extraction, and sentiment analysis.
* Machine Learning Libraries: Libraries like sklearn, joblib, pandas, seaborn are employed for training and evaluating machine learning models for sentiment classification.
* Flask: A Python web framework used for building the web application to showcase the sentiment analysis results.
* HTML/CSS/JavaScript: Front-end technologies used for designing and styling the web application user interface.
* MySQL: Databases to store the movie reviews and their sentiment labels.

## Project Structure
The project is structured as follows:

* data/: Contains the dataset files used for training and testing the sentiment analysis model.
* preprocessing/: Includes scripts or modules for text preprocessing tasks such as tokenization, stopword removal, and stemming.
* models/: Contains the implementation of the sentiment analysis models, including feature extraction, training, and evaluation.
* webapp/: Includes the code and templates for the web application that showcases the sentiment analysis results.
* requirements.txt: Lists all the Python dependencies required for running the project.

## How To Use
To run the Yoruba Movie Sentiment Analysis project, follow these steps:

* Clone the repository:
git clone https://github.com/myles4321/SentiYoruba.git
* Install the required dependencies using pip:
pip install -r requirements.txt
* Prepare the YOSM Corpus dataset by placing the movie reviews in the appropriate directory.
* Run the data preprocessing scripts to clean and preprocess the Yoruba movie reviews.
* Train and evaluate the sentiment analysis models using the preprocessed data.
* Build and run the web application to showcase the sentiment analysis results.
* Access the web application through a web browser to interact with the sentiment analysis system.



## Future Enhancements
The project can be further improved in the following ways:

* Implement advanced NLP techniques like word embeddings or transformer-based models to enhance sentiment classification accuracy.
* Integrate an automatic data collection module to retrieve real-time Yoruba movie reviews from online platforms.
* Enhance the web application with additional features such as word clouds, sentiment distribution charts, or user feedback submission.
* Deploy the sentiment analysis system on a cloud platform for wider accessibility and scalability.

## Contributions
Contributions to the project are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.
