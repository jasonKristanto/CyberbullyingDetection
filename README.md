# Cyberbullying Detection

This project contains web project that used for detecting wheter a text is cyberbullying or not. This project is built as a part of research conducted to detect cyberbullying texts. The algorithm used to construct this project is Long Short-Term Memory as a classification algorithm and Fasttext as word embedding. Programming language and web framework that is used to build this project is Python and Flask. This project contains detection model and Fasttext model for 100 and 300 Fasttext embedding size. 100 Fasttext embedding size can be chosen for device that has low memory while 300 Fasttext embedding size can be chosen for higher memory. Result for 300 Fasttext embedding size should give better result than the 100 embedding size. Nevertheless, the result of the 300 embedding size still not good because the score given in machine learning metric (precision, recall, and f-measure) is 72%, 69%, and 70%.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.


### Prerequisites

What things you need to install the software

* Anaconda - To run python project, and to store python library
* Keras and Tensorflow - To run classification
* Gensim library - To run Fasttext word embedding

Notes: Version can be seen on the Built With section

### Installing

1. Git clone this project  (https://github.com/jasonKristanto/CyberbullyingDetection.git)
2. Run the Anaconda Prompt which has been installed
3. Go to the directory that contains the project
4. Run the command
```
python app.py
```
5. Wait until URL link **127.0.0.1/5000** showed up on the Anaconda Prompt. If the web page doesn't showed up automatically, you can go to the URL link on your web browser
6. Detect your text!
7. You can detect a text on the form on the left, or some texts that are stored in CSV file on the form on the right
8. The result can be seen on the table below the form


## Authors

* **Jason Kristanto** - https://github.com/jasonKristanto

## Built With

* [Python 3.7.6](https://www.python.org/) - The programming language
* [Flask 1.1.1](https://flask.palletsprojects.com/en/1.1.x/) - The web framework
* [Keras 2.2.5](https://keras.io/) - The classification library
* [Tensorflow 1.14.0](https://www.tensorflow.org/) - The classification library
* [Gensim 3.8.0](https://radimrehurek.com/gensim/models/fasttext.html) - The Fasttext word embedding library
* [jQuery 3.4.1](https://jquery.com/download/) - The web front-end framework
* [Bootstrap](https://getbootstrap.com/) - The web front-end framework

## Acknowledgments

* ReadMe template provides by [PurpleBooth](https://github.com/PurpleBooth)
* Dataset used for train models by this [Conference Paper](https://www.researchgate.net/publication/322944989_Cyberbullying_comment_classification_on_Indonesian_Selebgram_using_support_vector_machine_method)
