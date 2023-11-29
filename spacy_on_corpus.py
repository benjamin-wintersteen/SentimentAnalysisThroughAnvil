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
#import pyate
import pyate
#import transformers
from transformers import pipeline
#import sentence transformers
from sentence_transformers import SentenceTransformer



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
        # Add each item in list_of_items to this counter (2 lines)
        for item in list_of_items:
            self.add_item(item)
        # HINT: Use the add_item method

        # Reduce to top k if top_k is greater than 0 (2 lines)
        if top_k > 0:
            self.reduce_to_top_k(top_k)
        # HINT: Use the reduce_to_top_k method
        # you don't have to return explicitly, since this is a constructor
        
    def add_item(self, item):
        """Adds an item to the counter.

        :param item: thing to add
        :type item: any
        """
        # COPY FROM PROJECT 3c
         # Add an item to this counter (3 lines)
        if item not in self:
            self[item] = 0
        self[item] += 1
        # HINT: use self[item], since a counter is a dictionary
        
    def get_counts(self):
        """Gets the counts from this counter.

        :returns: a list of (item, count) pairs
        :type item: list[tuple]
        """
        # COPY FROM PROJECT 3c
        return list(sorted(self.items(), key=lambda x:x[1]))
    
    def reduce_to_top_k(self, top_k):
        """Gets the top k most frequent items.

        :param top_k: the number you want to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
         # making sure we don't try to remove more elements than there are!
        top_k = min([top_k, len(self)])
        # Sort the frequency table by frequency (least to most frequent) (1 line)
        sorted_keys = sorted(self, key = lambda x: self[x])
        # Drop all but the top k from this counter (2 lines)
        # HINT: go from 0 to len(self)-top_k
        # HINT: use the pop() method; after all, counter is a dictionary!
        for i in range(0, len(self)- top_k):
            self.pop(sorted_keys[i])
    def pop(self, element):
        super().pop(element)
        #print('This method is redefined in Counter')

class corpus(dict):
    nlp = spacy.load('en_core_web_md')          
    nlp.add_pipe('combo_basic')
    classifier = pipeline('sentiment-analysis')
    summarizer = pipeline('summarization')
    # Pre-calculate embeddings; consider any embedding model 
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def __init__(self, name=''):
        """Creates or extends a corpus.

        :param name: the name of this corpus
        :type name: str
        :returns: a corpus
        :rtype: corpus
        """
        super().__init__()
        # COPY FROM PROJECT 3c
        self.name = name
       
    def get_documents(self):
        """Gets the documents from the corpus.

        :returns: a list of spaCy documents
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        return [item['doc'] for item in self.values()]
   
    def get_document(self, id):
        """Gets a document from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a spaCy document
        :rtype: (spaCy) doc
        """
        # COPY FROM PROJECT 3c
        return self[id]['doc'] if id in self and 'doc' in self[id] else None
    
    def get_document_texts(self):
        """Gets the document texts from the corpus.

        :returns: a spaCy document
        :rtype: (spaCy) doc
        """
        keys = list(self.keys())
        # get the text from each entry in the corpus
        text = {}
        for key in keys:
            text[key] = self.get_document(key)
        return list(text.items())
                         
    def get_metadatas(self):
        """Gets the metadata for each document from the corpus.

        :returns: a list of metadata dictionaries
        :rtype: list[dict]
        """
        # COPY FROM PROJECT 3c
        return [item['metadata'] for item in self.values()] # replace None

    def get_metadata(self, id):
        """Gets a metadata from the corpus.

        :param id: the document id to get
        :type id: str
        :returns: a metadata dictionary
        :rtype: dict
        """
        # COPY FROM PROJECT 3c
        return self[id]['metadata'] if id in self and 'metadata' in self[id] else None # replace None
    
    def get_sentiments(self):
        """Gets the metadata for each document from the corpus.

        :returns: a list of metadata dictionaries
        :rtype: list[dict]
        """
        # COPY FROM PROJECT 3c
        return [item['sentiment-analysis'] for item in self.values()] # replace None
              
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
        
        #add_pipeline = classifier(str(doc)) # , 'sentiment-analysis': add_pipeline
        self[id] = {'doc': self.nlp(doc), 'metadata': metadata, 'sentiment-analysis': corpus.classifier(str(doc))[0], 'summarization': corpus.summarizer(str(doc), min_length = 5, max_length = 20)[0]}
        
        
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
        # Make an empty list of tokens (1 line)
        tokens = []
        # For each doc in the corpus, add its tokens to the list of tokens (2 lines)
        for doc in self.get_documents():
            tokens.extend([token.text for token in doc if token.pos_ not in tags_to_exclude])
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        return counter(tokens, top_k = top_k).get_counts()
        # HINT: use the counter class

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
        # Make an empty list of entities (1 line)
        entities = []
        # For each doc in the corpus, add its entities to the list of entities (2 lines)
        for doc in self.get_documents():
            entities.extend([entity.text for entity in doc.ents if entity.label_ not in tags_to_exclude])
        # Count the entities using a counter object; return a list of pairs (item, frequency) (1 line)
        return counter(entities, top_k = top_k).get_counts()
        # HINT: use the counter class

    def get_noun_chunk_counts(self, top_k=-1):
        """Builds a noun chunk frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        # Make an empty list of chunks (1 line)
        chunks = []
        # For each doc in the corpus, add its chunks to the list of chunks (2 lines)
        for doc in self.get_documents():
            chunks.extend([chunk.text for chunk in doc.noun_chunks])
        # Count the chunks using a counter object; return a list of pairs (item, frequency) (1 line)
        return counter(chunks, top_k = top_k).get_counts()
        # HINT: use the counter class
    def get_sentiment_counts(self, top_k=-1):
        """Builds a sentiment frequency table.

        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # COPY FROM PROJECT 3c
        # Make an empty list of chunks (1 line)
        sentiments = []
        # For each doc in the corpus, add its chunks to the list of chunks (2 lines)
        sentiments.extend([sentiment['label'] for sentiment in self.get_sentiments()])
        # Count the chunks using a counter object; return a list of pairs (item, frequency) (1 line)
        return counter(sentiments, top_k = top_k).get_counts()
        # HINT: use the counter class

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
        # Using get_token_counts as a model, define get_metadata_counts using get_metadatas and a counter object (2 lines of code)
        metadatas = []
        metadatas.extend([metadata[key] for metadata in self.get_metadatas() if key in metadata ]) 
        # HINT: use a list comprehension, the get_metadatas method, and then the counter class

        # return the metadata counts as a list of pairs
        return counter(metadatas, top_k = top_k).get_counts()

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
        entity_counts = self.get_entity_counts()
        text = f'Entities: %i\n' % sum([x[1] for x in entity_counts])
        text += f"Unique Entities: %i\n" % len(entity_counts)
        return text
        
    def get_noun_chunk_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        noun_chunk_counts = self.get_noun_chunk_counts()
        text = f'Noun chunks: %i\n' % sum([x[1] for x in noun_chunk_counts])
        text += f"Unique Noun chunks: %i\n" % len(noun_chunk_counts)
        return text
    def get_sentiment_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        sentiment_counts = self.get_sentiment_counts()
        text = f'Positive Documents: %i\n' % sum([x[1] for x in sentiment_counts if x[0] == "POSITIVE"])
        text += f"Neutral Docuements: %i\n" % sum([x[1] for x in sentiment_counts if x[0] == "NEUTRAL"])
        text += f"Negative Docuements: %i\n" % sum([x[1] for x in sentiment_counts if x[0] == "NEGATIVE"])
        return text
    
    def get_basic_statistics(self):
        """Prints summary statistics for the corpus.
        
        :returns: the statistics report
        :rtype: str
        """
        # FOR PROJECT 4a: make this use get_token_statistics, get_entity_statistics and get_noun_chunk_statistics; also, instead of printing, return as a string.
        text = ''
        text += self.get_token_statistics()
        text += self.get_entity_statistics()
        text += self.get_noun_chunk_statistics()
        text += self.get_sentiment_statistics()
        return text

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
        token_counts = self.get_token_counts()
        self.plot_counts( token_counts, 'token_frequencies.png')

    def plot_entity_frequencies(self, tags_to_exclude=['QUANTITY'], top_k=25):
        """Makes a bar chart for the top k most frequent entities in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
       """
        # COPY FROM PROJECT 3c
        self.plot_counts(self.get_entity_counts(tags_to_exclude=tags_to_exclude, top_k = top_k), 'entity_frequencies.png')

     
    def plot_noun_chunk_frequencies(self, top_k=25):
        """Makes a bar chart for the top k most frequent noun chunks in the corpus.
        
        :param top_k: the number to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
        self.plot_counts(self.get_noun_chunk_counts(top_k = top_k), 'chunk_frequencies.png')
     
    def plot_metadata_frequencies(self, key, top_k=25):
        """Makes a bar chart for the frequencies of values of a metadata key in a corpus.

        :param key: a metadata key
        :type key: str        
        :param top_k: the number to keep
        :type top_k: int
        """
        # COPY FROM PROJECT 3c
        self.plot_counts(self.get_metadata_counts(key, top_k = top_k), str(key) + '_frequencies.png')
 
    def plot_word_cloud(self, counts, file_name):
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
        return self.plot_word_cloud(self.get_token_counts(tags_to_exclude=tags_to_exclude), 'token_cloud.png')
 
    def plot_entity_cloud(self, tags_to_exclude=['QUANTITY']):
        """Makes a word cloud for the frequencies of entities in a corpus.
 
        :param tags_to_exclude: tags to exclude
        :type tags_to_exclude: list[str]
        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        return self.plot_word_cloud(self.get_entity_counts(tags_to_exclude=tags_to_exclude), 'entity_cloud.png')

    def plot_noun_chunk_cloud(self):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        return self.plot_word_cloud(self.get_noun_chunk_counts(), 'chunk_cloud.png')
    
    def plot_sentiment_cloud(self):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloud
        """
        # COPY FROM PROJECT 3c, then add return value
        return self.plot_word_cloud(self.get_sentiment_counts(), 'sentiment_cloud.png')
        
    def update_document_metadata(self, id, value_key_pair):
        """Makes a word cloud for the frequencies of noun chunks in a corpus.

        :returns: the word cloud
        :rtype: wordcloud
        """
        #try:
        for i in value_key_pair.keys():
            self[id]['metadata'][i] = value_key_pair[i]
        #except: 
        #    print('id not in dictionary')

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
        for token in doc:
        # if the token is a noun, add it to 'text' and make it boldface (HTML: <b> at the start, </b> at the end)
            if token.pos_ == 'NOUN':
                text = text + '**' + token.text + '**'
        # otherwise, if it's a verb, add it to 'text' and make it italicized (HTML: <i> at the start, </i> at the end)
            elif token.pos_ == 'VERB':
                text = text + '*' + token.text + '*'
        # otherwise, just add it to 'text'!
            else:
                text = text + token.text
        # add any whitespace following the token using attribute whitespace_
            text = text + token.whitespace_

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
        for token in doc:
            # if the token's part of speech is not 'SPACE'
            if token.pos_ != 'SPACE':
                # add the the text, lemma, coarse- and fine-grained parts of speech, word shape and morphology for this token to `tokens_table`
                tokens_table  = tokens_table + "| " + token.text + " | " + token.lemma_ + " | " + token.pos_ + " | " + token.tag_ + " | " + token.shape_ + " | " + re.sub(r'\|', '#', str(token.morph)) + " |\n"
        # make the entities table
        entities_table = "| Text | Type |\n| ---- | ---- |\n"
        # walk over the entities in the document
        for entity in doc.ents:
            # add the text and label for this entity to 'entities_table'
            entities_table = entities_table + "| " + entity.text + " | " + entity.label_ + " |\n"
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
        stats = {}
        # walk over the tokens in the document
        for token in doc:
            # if the token's part of speech is not 'SPACE'
            if token.pos_ != 'SPACE':
                # add the token and its part of speech tag ('pos_') to 'stats' (check if it is in 'stats' first!)
                if token.text + token.pos_ not in stats:
                    stats[token.text + token.pos_] = 0
                # increment its count
                stats[token.text + token.pos_] = stats[token.text + token.pos_] + 1
        # walk over the entities in the document
        for entity in doc.ents:
            # add the entity and its label ('label_') to 'stat's (check if it is in 'stat's first!)
            if entity.text + entity.label_ not in stats:
                stats[entity.text + entity.label_] = 0
            # increment its count
            stats[entity.text + entity.label_] = stats[entity.text + entity.label_] + 1
        #walk over the noun chunks in the document
        for chunk in doc.noun_chunks:
            # add the entity and its label ('label_') to 'stat's (check if it is in 'stat's first!)
            if chunk.text + chunk.label_ not in stats:
                stats[chunk.text + chunk.label_] = 0
            stats[chunk.text + chunk.label_] += 1
        
        for sentence in doc.sents:
            # adds unique sentences to stats
            if sentence.text not in stats:
                stats[sentence.text] = 0
            stats[sentence.text] += 1
        text = '| Token/Entity | Count |\n | ------------ | ----- |\n'
        for key in sorted(stats.keys()):
                text += ('| ' + str(key) + ' | ' + str(stats[key]) + ' |\n')
        # print the key and count for each entry in 'stats'


        return text
    def render_doc_summary(self, doc_id):
        return self[doc_id]['summarization']['summary_text']
    
    def get_keyphrase_counts(self, top_k = -1):
        """Builds a keyphrase frequency table.

        :param tags_to_exclude: (Coarse-grained) part of speech tags to exclude from the results
        :type tags_to_exclude: list[string]
        :param top_k: how many to keep
        :type top_k: int
        :returns: a list of pairs (item, frequency)
        :rtype: list
        """
        # Make an empty list of tokens (1 line)
        keyphrases = []
        # For each doc in the corpus, add its tokens to the list of tokens (2 lines)
        for doc in self.get_documents():
            keyphrases.extend(list(doc._.combo_basic.keys()))
        # Count the tokens using a counter object; return a list of pairs (item, frequency) (1 line)
        return counter(keyphrases, top_k = top_k).get_counts()
        # HINT: use the counter class
    def get_keyphrase_statistics(self):
        """Prints summary statistics for noun chunks in the corpus. Model on get_token_statistics.
        
        :returns: the statistics report
        :rtype: str
        """
        # NEW FOR PROJECT 4a
        keyphrase_counts = self.get_keyphrase_counts()
        text = f'Keyphrases: %i\n' % sum([x[1] for x in keyphrase_counts])
        text += f"Unique Keyphrases: %i\n" % len(keyphrase_counts)
        return text
    def build_topic_model(self):
        full_texts = [self[x]['doc'].text for x in self]*50
        embeddings = corpus.embedding_model.encode(full_texts, show_progress_bar=True)

        # Prevent stochastic behavior
        from umap import UMAP

        # choose a number of neighbors that's reasonable for your data set
        umap_model = UMAP(n_neighbors=5, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

        # Set minimum cluster size
        from hdbscan import HDBSCAN

        # choose a minimum cluster size that's reasonable for your data set
        hdbscan_model = HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # I find this dubious but okay; "Improve" default representation
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
        # Use multiple representations
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech

        # KeyBERT
        keybert_model = KeyBERTInspired()

        # Part-of-Speech
        pos_model = PartOfSpeech("en_core_web_md")

        # MMR
        mmr_model = MaximalMarginalRelevance(diversity=0.3)

        # All representation models
        representation_model = {
            "KeyBERT": keybert_model,
            "MMR": mmr_model,
            "POS": pos_model
        }
        # Make topic model using all of this setup
        from bertopic import BERTopic

        topic_model = BERTopic(

        # Pipeline models
        embedding_model=corpus.embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        representation_model=representation_model,

        # Hyperparameters
        top_n_words=10,
        verbose=True,
        nr_topics="auto"
        )
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
        # make sure we have a corpus
        if my_corpus == None:
            my_corpus = corpus()
        # Mostly the same as in project 3b, but use the corpus class add method; don't forget to return my_corpus (3 lines of code)
        # open file_name as f
        with open(file_name) as f:   
        # make a dictionary containing the key 'doc' mapped to the spacy document for the text in file_name; then in the 'corpus' dictionary add the key file_name and map it to this dictionary
        #corpus.update({file_name: new_dict})
            my_corpus.add_document(file_name, cls.nlp(' '.join(f.readlines())))
        return my_corpus

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
        # make sure we have a my_corpus
        if my_corpus == None:
            my_corpus = corpus()
        # Most the same as in project 3b, but use the corpus add method; don't forget to return my_corpus (6 lines of code)
        with open(file_name, encoding='utf-8') as f:
        # walk over all the lines in the file
            for line in f.readlines():
                # load the python dictionary from the line using the json package; assign the result to the variabe 'js'
                js = json.loads(line)
                # if there are keys 'id' and 'fullText' in 'js'
                if 'id' in js.keys() and 'fullText' in js.keys():
                    my_corpus.add_document(js['id'], cls.nlp(''.join(js["fullText"])), metadata = js)
        return my_corpus

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
        # make sure we have a corpus
        if my_corpus == None:
            my_corpus = corpus()
        # Mostly the same as in project 3b; don't forget to return my_corpus (5 lines of code)
        # uncompress the compressed file
        shutil.unpack_archive(file_name, 'temp')
        # for each file_name in the compressed file
        for file_name2 in glob.glob('temp/*'):
        # build the corpus using the contents of file_name2 
            my_corpus.build_corpus(file_name2, my_corpus = my_corpus)

        # clean up by removing the extracted files
        shutil.rmtree("temp")
        return my_corpus

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
         # make sure we have a corpus
        if my_corpus == None:
            my_corpus = corpus(pattern)
       # Mostly the same as in project 3b; don't forget to return my_corpus (11 lines of code)
        try:
                # for each file_name matching pattern
                for file_name in glob.glob(pattern):
                    # if file_name ends with '.zip', '.tar' or '.tgz' # to actually find end would need to convert this to string
                    # if [file_name[len(str(file_name)) - (i+1)] for i in range(4)] == '.zip' or '.tar' or '.tgz':
                    if file_name.endswith('.zip') or file_name.endswith('.tar') or file_name.endswith('.tgz'):

                        # then call load_compressed
                        cls.load_compressed(file_name, my_corpus)
                    # if file_name ends with '.jsonl'
                    elif file_name.endswith('.jsonl'):
                        # then call load_jsonl
                        cls.load_jsonl(file_name, my_corpus)
                    # otherwise (we assume the files are just text)
                    else:
                        # then call load_textfile
                        cls.load_textfile(file_name, my_corpus) 

        except Exception as e: # if it doesn't work, say why
            print(f"Couldn't load % s due to error %s" % (pattern, str(e)))
            # return the corpus
        return my_corpus
