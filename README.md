[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/Mng_pNG3)
# Analyzing Corpora with Spacy - The Web App

# Project 4a

## Name
Benjamin Wintersteen
## Class year
2027
## Extension(s) - Describe your extension(s) here
Not extensions but here are the the descriptions and aspects of my code complete functionalities:
My add document and clear document work. My build corpus is not updating the corpus. This means all the other functionalities are based off the text from the test.jsonl file

Here is my implementation of token_counts on test.jsonl:
![Alt text](image.png)
Here is the start of my implementation of document table on the first doc in test.jsonl
![Alt text](image-1.png)
## Resources - Who or what did you use to finish this project deliverable?

-----------------------------------------------------------------------------------------------------------------------------------------------
*Please do not edit below this line!*
-----------------------------------------------------------------------------------------------------------------------------------------------

## Project Description

The goals of this assignment are:

* To work with the object oriented version of our corpus code.
* To make a web app (!) that we can use to analyze text data.

Here, **in order**, are the steps you should do to successfully complete this project:

1. From moodle, accept the assignment. Open and set up a code space (install a python kernel and select it).
2. I wrote the comments; you write the code! Modify `spacy_on_corpus.py` following the instructions in this notebook.
3. Complete the notebook and commit it to Github. Make sure to commit the notebook in a "run" state!
4. Edit the README.md file. Provide your name, your class year, links to/descriptions of any extensions and a list of resources. 
5. Commit your code often. We will take the last commit before the deadline as your submission of the project.

Possible extensions (from least points to most points):

* Modify the token, entity, and noun chunk get count methods so they count only lower cased tokens, entities and noun chunks.
* Modify the [styling](https://anvil.works/learn/tutorials/using-material-3) of the web app. 
* To the screen `Build Corpus` in the web app, add the ability for the user to choose the language of their input documents.
* To the screen `Analyze Document` in the web app, add the ability for the user to choose a value for `top_k` and to choose which token and entity tags to exclude.
* Plot more than one analysis at a time in `Analyze Corpus` (see [this page](https://anvil.works/docs/client/components/plots)).
* Add the ability for a user to enter jsonl in the input text area on the `Build Corpus` screen.
* If you added paragraphs to project 3c, port that over to project 4a.
* Add some metadata analysis and visualization on a fourth screen.
* Your other ideas are welcome! If you'd like to discuss one with Dr Stent, feel free.

## Project Rubric

- [3] Notebook is code-complete and outputs are correct for the three listed functionalities. (4 points)
- [4] File spacy_on_corpus.py is complete, runs and is commented for the three listed functionalities. (4  points)
- [1] Student made meaningful commit messages (1 pt)
- [1] Readme has student's name, class year and resources student used. (1 pt)


### Comments from grader

* Only have 2 functionalities shown in the Readme + screenshots in the repo
