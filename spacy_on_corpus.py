# import spacy for nlp
import spacy
# import glob in case user enters a file pattern
import glob
# import shutil in case user enters a compressed archive (.zip, .tar, .tgz etc.); this is more general than zipfile
import shutil
# import plotly for making graphs
import plotly.express as px
# import wordcloud for making wordclouds
import wordcloud
# import json
import json 
# import re
import re


class counter(dict):
    def __init__(self, list_of_items, top_k=-1):
        """Makes a counter.

        :param list_of_items: the items to count
        :type list_of_items: list
        :param top_k: the number you want to keep
        :type top_k: int
        :returns: a counter
        :rtype: counter
        """
        super().__init__()
        # COPY FROM PROJECT 3c

        # you don't have to return explicitly, since this is a constructor
        
    def add_item(self, item):
        """Adds an item to the counter.

        :param item: thing to add
        :type item: any
        """
        # COPY FROM PROJECT 3c
        pass
        
    def get_counts(self):
        """Gets the counts from this counter.

        :returns: a list of (item, count) pairs
        :type item: list[tuple]
        """
        # COPY FROM PROJECT 3c
        pass
    
    def reduce_to_top_k(self, top_k):
        """Gets the top k most frequent items.

        :param top_k: the number you want to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
        pass

class corpus(dict):
    nlp = spacy.load('en_core_web_md')          
                         
    def __init__(self, name=''):
        """Creates or extends a corpus.

        :param name: the name of this corpus
        :type name: str
        :returns: a corpus
        :rtype: corpus
        """
        super().__init__()
        # COPY FROM PROJECT 3c
       
    def get_documents(self):
        """Gets the documents from the corpus.

        :returns: a list of spaCy documents
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        pass
   
    def get_document(self, id):
        """Gets a document from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a spaCy document
        :rtype: (spaCy) doc
        """
        # COPY FROM PROJECT 3c
        pass
                         
    def get_metadatas(self):
        """Gets the metadata for each document from the corpus.

        :returns: a list of metadata dictionaries
        :rtype: list[dict]
        """
        # COPY FROM PROJECT 3c
        pass

    def get_metadata(self, id):
        """Gets a metadata from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a metadata dictionary
        :rtype: dict
        """
        # COPY FROM PROJECT 3c
        pass
                         
    def add_document(self, id, doc, metadata={}):
        """Adds a document to the corpus.

        :param id: the document id
        :type id: str
        :param doc: the document itself
        :type doc: (spaCy) doc
        :param metadata: the document metadata
        :type metadata: dict
        """
        # COPY FROM PROJECT 3c
        pass
        
    def get_token_counts(self, tags_to_exclude = ['PUNCT', 'SPACE'], top_k=-1):
        """Builds a token frequency table.

        :param tags_to_exclude: (Coarse-grained) part of speech tags to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        pass

    def get_entity_counts(self, tags_to_exclude = ['QUANTITY'], top_k=-1):
        """Builds an entity frequency table.

        :param tags_to_exclude: named entity labels to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        pass

    def get_noun_chunk_counts(self, top_k=-1):
        """Builds a noun chunk frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        pass

    def get_metadata_counts(self, key, top_k=-1):
        """Gets frequency data for the values of a particular metadata key.

        :param key: a key in the metadata dictionary
        :type key: str
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        pass

    def get_token_statistics(self):
        """Prints summary statistics for tokens in the corpus, including: number of documents; number of sentences; number of tokens; number of unique tokens.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        text = f'Documents: %i\n' % len(self)
        text += f'Sentences: %i\n' % sum([len(list(doc.sents)) for doc in self.get_documents()])
        token_counts = self.get_token_counts()
        text += f'Tokens: %i\n' % sum([x[1] for x in token_counts])
        text += f"Unique tokens: %i\n" % len(token_counts)
        return text

    def get_entity_statistics(self):
        """Prints summary statistics for entities in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        return ''
        
    def get_noun_chunk_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        return ''
    
    def get_basic_statistics(self):
        """Prints summary statistics for the corpus.
        
        :returns: the statistics report
        :rtype: str
        """
        # FOR PROJECT 4a: make this use get_token_statistics, get_entity_statistics and get_noun_chunk_statistics; also, instead of printing, return as a string.
        return ''

    def plot_counts(self, counts, file_name):
        """Makes a bar chart for counts.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        """
        fig = px.bar(x=[x[0] for x in counts], y=[x[1] for x in counts])
        fig.write_image(file_name)

    def plot_token_frequencies(self, tags_to_exclude=['PUNCT', 'SPACE'], top_k=25):
        """Makes a bar chart for the top k most frequent tokens in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        """
        # COPY FROM PROJECT 3c
        pass

    def plot_entity_frequencies(self, tags_to_exclude=['QUANTITY'], top_k=25):
        """Makes a bar chart for the top k most frequent entities in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
       """
        # COPY FROM PROJECT 3c
        pass
     
    def plot_noun_chunk_frequencies(self, top_k=25):
        """Makes a bar chart for the top k most frequent noun chunks in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
        pass
     
    def plot_metadata_frequencies(self, key, top_k=25):
        """Makes a bar chart for the frequencies of values of a metadata key in a corpus.

        :param key: a metadata key
        :type key: str        
        :param top_k: the number to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
        pass
 
    def plot_word_cloud(self, name, counts):
        """Plots a word cloud.

        :param counts: a list of item, count tuples
        :type counts: list
        :param file_name: where to save the plot
        :type file_name: string
        :returns: the word cloud
        :rtype: wordcloud
        """
        wc = wordcloud.WordCloud(width=800, height=400, max_words=200).generate_from_frequencies(dict(counts))
        cloud = px.imshow(wc)
        cloud.update_xaxes(showticklabels=False)
        cloud.update_yaxes(showticklabels=False)
        return cloud

    def plot_token_cloud(self, tags_to_exclude=['PUNCT', 'SPACE']):
        """Makes a word cloud for the frequencies of tokens in a corpus.

        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        pass
 
    def plot_entity_cloud(self, tags_to_exclude=['QUANTITY']):
        """Makes a word cloud for the frequencies of entities in a corpus.
 
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        pass

    def plot_noun_chunk_cloud(self):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        pass
        
    def render_doc_markdown(self, doc_id):
        """Render a document as markdown. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str

        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # define 'text' and set the title to be the document key (file name)
        text = '# ' + doc_id + '\n\n'
        # walk over the tokens in the document

        return text

    def render_doc_table(self, doc_id):
        """Render a document's token and entity annotations as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # make the tokens table
        tokens_table = "| Tokens | Lemmas | Coarse | Fine | Shapes | Morphology |\n| ------ | ------ | ------ | ---- | ------ | ---------- | \n"
        # walk over the tokens in the document

        # make the entities table
        entities_table = "| Text | Type |\n| ---- | ---- |\n"
        # walk over the entities in the document
 
        return '## Tokens\n' + tokens_table + '\n## Entities\n' + entities_table

    def render_doc_statistics(self, doc_id):
        """Render a document's token and entity counts as a table. From project 2a. 

        :param doc_id: the id of a spaCy doc made from the text in the document
        :type doc: str
        :returns: the markdown
        :rtype: str
        """
        # MODIFIED FROM PROJECT 3c: instead of printing or saving the markdown, return it as a string
        doc = self.get_document(doc_id)
        # Same definition as in project 3b, but prefix the output file name with self.name to make it unique to this corpus
        # make a dictionary for the statistics

        text = '| Token/Entity | Count |\n | ------------ | ----- |\n'
        # print the key and count for each entry in 'stats'

        return text

    @classmethod
    def load_textfile(cls, file_name, my_corpus=None):
        """Loads a textfile into a corpus.

        :param file_name: the path to a text file
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
         """
        # COPY FROM PROJECT #c
        pass

    @classmethod  
    def load_jsonl(cls, file_name, my_corpus=None):
        """Loads a jsonl file into a corpus.

        :param file_name: the path to a jsonl file
        :type file_name: string
        :param my_corpus: a my_corpus
        :type my_corpus: my_corpus
        :returns: a my_corpus
        :rtype: my_corpus
         """
        # COPY FROM PROJECT #c
        pass

    @classmethod   
    def load_compressed(cls, file_name, my_corpus=None):
        """Loads a zipfile into a corpus.

        :param file_name: the path to a zipfile
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
       """
        # COPY FROM PROJECT #c
        pass

    @classmethod
    def build_corpus(cls, pattern, my_corpus=None):
        """Builds a corpus from a pattern that matches one or more compressed or text files.

        :param pattern: the pattern to match to find files to add to the corpus
        :type file_name: string
        :param my_corpus: a corpus
        :type my_corpus: corpus
        :returns: a corpus
        :rtype: corpus
         """
        # COPY FROM PROJECT #c
        pass
