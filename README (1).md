# Analyzing Corpora with Spacy - The Corpus Class

# Project 3c

## Name
benjamin Wintersteen
## Class year
2027
## Extension(s) - Describe your extension(s) here

## Resources - Who or what did you use to finish this project deliverable?
I worked with Vishnu and Ben K-B. I did a touch of debugging with ChatGPT and also used huggingface api. 

https://anvil.works/docs/client/components/plots
https://anvil.works/docs/client/components/plots
https://anvil.works/docs/client/components/basic#fileloader
https://anvil.works/docs/uplink/quickstart

extensions:
language 
color etc
top_k and tags to exclude x 2
plot more than one at a time (https://anvil.works/docs/client/components/plots)
add jsonl via input text area

1) move plot functions to anvil client side
2) client side should allow user to type text or upload a file, then to choose any of the things main permits - and render them

-----------------------------------------------------------------------------------------------------------------------------------------------
*Please do not edit below this line!*
-----------------------------------------------------------------------------------------------------------------------------------------------

## Project Description

The goals of this assignment are:
* To develop an object oriented version of our corpus code, focusing on writing the most concise readable code possible.

Here are the steps you should do to successfully complete this project:
1. From moodle, accept the assignment. Open and set up a code space (install a python kernel and select it).
2. Complete the notebook and commit it to Github. Make sure to answer all questions, and to commit the notebook in a "run" state!
3. I wrote the comments; you write the code! Complete and run `spacy_on_corpus.py` following the instructions in this notebook.
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

## Project 4a Rubric

- [] Notebook is code-complete and outputs are correct for the three listed functionalities. (4 points)
- [] File spacy_on_corpus.py is complete, runs and is commented for the three listed functionalities (including screenshots). (4  points)
- [] Student made meaningful commit messages (1 pt)
- [] Readme has student's name, class year and resources student used. (1 pt)

## Project 4b Rubric

- [] File spacy_on_corpus.py is complete and is commented. (6 points)
- [] Notebook is code-complete. (6 points)
- [] All requested screenshots are provided in notebook. (6 points)
- [] All ten questions are answered. (5 points)
- [] Student made meaningful commit messages. (1 pt)
- [] Readme has student's name, class year and resources student used. (1 pt)
- [] Extension (1-2 points for a start; 3-4 points for a complete extension; 5 points for a surprising and creative extension)

### Comments from grader
