import requests
import json
import spacy
from nltk.corpus import stopwords
from xml.etree import ElementTree as ET
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from openai import OpenAI
from helper.config import api_key
from sklearn.metrics.pairwise import cosine_similarity

# -------- LOAD IMPORTANT VARIABLES --------
# Load SpaCy's language model
nlp = spacy.load("en_core_web_sm")
# Define stop words using NLTK
stop_words = set(stopwords.words('english'))
with open('helper/descriptors.json') as json_file:
    descriptors = json.load(json_file)
with open('helper/concepts.json') as json_file:
    concepts = json.load(json_file)
with open('helper/terms.json') as json_file:
    terms = json.load(json_file)



# --------------------------------------------------------
#  MAJOR MANAGER CLASS THAT CALLS ON OTHER FUNCTIONS  
# --------------------------------------------------------

class QueryExpansionManager():
    def __init__(self, model_name, descriptor_json_location, api_key=None,) -> None:
        if model_name == 'openai':
            if api_key is None:
                raise ValueError("API key is required for OpenAI models")
            self.model_name = model_name
            self.client = OpenAI(api_key = api_key)
            self.tokenizer = None
            self.model = None
        else:
            self.model_name = model_name
            self.client = None
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        with open(descriptor_json_location) as json_file:
            self.descriptors = json.load(json_file)
        self.query = ''
        self.translated_query=''
        self.retrieved_article_details = {}    

    # This function handles embedding for all the rest of the task in this class 
    # It is crucial in standardizing the Manager class regardless of embedding models since it returns a unified format that can be used by all other functions.
    def embed_mesh_headings_preloaded_model(self, text):
        '''Takes in a text as argument and outputs the embedding vector.'''
        #if using an openai model
        if self.model_name == 'openai':
            text = text.replace("\n", " ")
            embeddings = np.array([self.client.embeddings.create(input = [text], model="text-embedding-3-large").data[0].embedding])
        else: 
            # Tokenize the text and get embeddings
            inputs = self.tokenizer([text], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = self.model(**inputs).last_hidden_state.mean(dim=1).numpy()  # Use mean pooling for sentence embedding            
        return embeddings    
    
    # Define search
    def get_retrieved_article_details_from_query(self, query, requery = False, remove_stop_words = False):
        # if not requery:
        if not requery:
            self.query = query
        self.retrieved_article_details, self.translated_query = get_retrieved_article_details_from_query(query, remove_stop_words=remove_stop_words)        

    # This function will extract
    # def get_filtered_entries_from_retrieved_article_details(self, need_heading_filter, need_article_filter, title_abstract_threshold, heading_threshold):
    def get_entries_from_retrieved_article_details(self):
        # '''Takes in several arguments and returns the filtered heading entries and article entries, in that order.'''
        '''Takes in the retrieved_article_details nested dictionary and returns the heading entries and article entries, in that order.'''
        # self.retrieved_article_details = self.retrieved_article_details
        # self.query = self.query
        # print('Query being embedded is: ', self.query)
        query_embedding = self.embed_mesh_headings_preloaded_model(self.query)
        filtered_heading_terms = []
        filtered_articles = []
        for key, value in self.retrieved_article_details.items():
            article_headings = []
            # Extract and embed all extracted headings for separate relevance ranking of the terms itself
            for mesh_heading in value['Mesh Headings']: 
                article_headings.append(mesh_heading['Descriptor'])
                mesh_heading_embeddings = self.embed_mesh_headings_preloaded_model(descriptors[mesh_heading['Descriptor']]['Combined Description'])   
                similarity = cosine_similarity(query_embedding, mesh_heading_embeddings)[0][0]    # output is in fraction
                # This is the part that controls whether heading details are outputed
                # If the similarity is less than threshold and need filter is enabled, it is not outputted
                # if (similarity < heading_threshold) and need_heading_filter:  
                #     continue
                # else:      
                heading_entry = {
                    'name': mesh_heading['Descriptor'],
                    'content': descriptors[mesh_heading['Descriptor']]['Combined Description'],
                    # 'embeddings': mesh_heading_embeddings,  # can uncomment for debugging/more details, not needed otherwise
                    'suitability': similarity*100
                }
                filtered_heading_terms.append(heading_entry)
            title = value['Title'] or ''
            abstract = value ['Abstract'] or ''
            title_abstract = title + abstract
            article_title_abstract_embeddings = self.embed_mesh_headings_preloaded_model(title_abstract)
            article_similarity = cosine_similarity(query_embedding, article_title_abstract_embeddings)[0][0]
            # link = f'https://pubmed.ncbi.nlm.nih.gov/{key}/'
            link = f'https://www.ncbi.nlm.nih.gov/pmc/articles/pmid/{key}/'
            # if (article_similarity < title_abstract_threshold) and need_article_filter:
            #     continue
            # else:
            article_entry = {
                'name': key,
                'content': title_abstract,
                'title': title,
                'abstract': abstract,
                # 'embeddings': article_title_abstract_embeddings,  # can uncomment for debugging/more details, not needed otherwise
                'headings': article_headings,
                'suitability': article_similarity*100,
                'link': link
            }
            filtered_articles.append(article_entry)
        return filtered_heading_terms, filtered_articles
    
    # This is a method that packages an external function in this page and a class-only method to create an end-to-end pipeline
    # Said pipeline takes in queries and returns filtered entries that can be shown on a page to users.
    # def get_filtered_entries_from_query (self, query, requery = False, articles_to_retrieve = 10, remove_stop_words = False, title_abstract_threshold = 0.5, heading_threshold = 0.5):
    def get_entries_from_query (self, query, requery = False, articles_to_retrieve = 10, remove_stop_words = False, **kwargs)-> list|list:

        '''
        Takes in 1) queries and associated arguments and 2) filtering arguments as kwargs and returns a list of filtered heading and article dictionaries respectively. \n
        Heading dictionaries contain name, content, embeddings and suitability. Article dictionaries contain name, content, embeddings, headings and suitability.         
        '''
        if not requery:
            self.query = query
        self.retrieved_article_details, self.translated_query = get_retrieved_article_details_from_query(query, articles_to_retrieve, remove_stop_words)
        # filtered_heading_terms, filtered_articles = self.get_filtered_entries_from_retrieved_article_details (need_heading_filter=kwargs['need_heading_filter'],
        #                                                                                                       need_article_filter=kwargs['need_article_filter'],
        #                                                                                                       heading_threshold=kwargs['heading_threshold'],
        #                                                                                                       title_abstract_threshold=kwargs['title_abstract_threshold'])
        heading_terms, article_title_abstracts = self.get_entries_from_retrieved_article_details()
        # return filtered_heading_terms, filtered_articles
        return heading_terms, article_title_abstracts
    
    # # I want this function to still be used outside of the class, so heading and article entries must be manually passed in
    # def create_semantic_neighbourhood_query(self, heading_entries:list, article_entries:list):
    #     '''
    #     Takes in one list of filtered heading and article entries each and creates a Boolean requery term out of it.
    #     '''
    #     # Extracts out the mesh heading terms that are found in the shortlisted 'heading entries'
    #     filtered_headings = [x['name'] for x in heading_entries]
    #     semantic_neighbourhoods = []
    #     for article_entry in article_entries:
    #         expansion_set = article_entry['headings']
    #         if len(expansion_set)!=0:
    #             chunk_fragments_to_expand = []
    #             for i, val in enumerate(expansion_set):
    #                 # First, vet out mesh terms deemed not relevant to the search
    #                 if val not in filtered_headings:
    #                     continue
    #                 if i == 0: # first term of set
    #                     chunk_fragments_to_expand.append(f'("{val}"[MeSH Terms]')
    #                 # elif i == (len(expansion_set)-1): # final term of set
    #                 #     chunk_fragments_to_expand.append(f'AND "{val}"[MeSH Terms])')
    #                 else:
    #                     chunk_fragments_to_expand.append(f'AND "{val}"[MeSH Terms]')
    #             if len(chunk_fragments_to_expand)!=0: # 0 will happen if all the fragments are not relevant semantically
    #                 chunk_fragments_to_expand.append(')')
    #                 article_semantic_neighbourhood = ' '.join(chunk_fragments_to_expand)
    #             # if article_semantic_neighbourhood!= '':
    #                 semantic_neighbourhoods.append(article_semantic_neighbourhood)
    #     if semantic_neighbourhoods != []:
    #         query_with_semantic_neighbourhoods = " OR ".join(semantic_neighbourhoods)
    #     else:
    #         query_with_semantic_neighbourhoods = ''
    #     return query_with_semantic_neighbourhoods







# ------------------------------
#  AUXILLIARY SUPPORT FUNCTIONS  
# ------------------------------

def remove_stop_words(query, pos_to_maintain = ["NOUN", "ADJ", "VERB"]):
    ''' This function removes all words from a query except those specified in the post_to_maintain argument, then returns the cleansed query.'''
    doc = nlp(query)
    # Filter out stop words and only keep nouns and adjectives
    filtered_tokens = [token.text for token in doc if token.pos_ in pos_to_maintain and token.text.lower() not in stop_words]
    return " ".join(filtered_tokens)


def get_query_response(query, remove_stop_words = False, articles_to_retrieve = 10):
    '''This is the workhorse of the application. It receives a query, queries PubMed and then sends PubMed's response back to the user.'''
    if remove_stop_words:
        query = remove_stop_words(query)
        print('Preprocessed query:', query)
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",        # Database to search
        "term": query,          # Natural language query
        "retmax": articles_to_retrieve,         # Number of results to return
        'api_key' : '2016449eab49266b3ccf1be9ba0c52b8d809',
        "retmode": "json",      # Response format    
        'maxdate':"2023/06/30"
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
    else:
        print(f"Error: {response.status_code}")
    return data


def get_article_details_from_id(id_list, important_headings_only = True):
    '''Takes in a list of article ids and outputs a dictionary where the keys are article ids and the values are article details.\n
        The article details include Title, Abstract and Mesh Headings. '''
    
    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),        # Join PMIDs with commas
        "retmode": "xml",               # Request XML format for easier parsing
        "rettype": "abstract"           # Request abstract type
    }
    # Send the EFetch request
    efetch_response = requests.get(efetch_url, params=efetch_params)    
    if efetch_response.status_code == 200:
        try:
            root = ET.fromstring(efetch_response.content)
            articles = root.findall(".//PubmedArticle")
            # This is the part that parses the returned xml to find the most relevant details for each PMID.
            retrieved_data = {}
            for article in articles:
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle")
                abstract = article.findtext(".//AbstractText")            
                # Extract all MeSH Headings and details for each article
                mesh_headings = []
                for mesh_heading in article.findall(".//MeshHeading/DescriptorName"):
                    descriptor_name = mesh_heading.text
                    descriptor_ui = mesh_heading.attrib['UI']     
                    descriptor_major_topic = mesh_heading.attrib.get('MajorTopicYN', 'Not Found')
                    # This one will extract only major article descriptors if important_headings_only is set to True
                    if (important_headings_only and (descriptor_major_topic=='Y')) or not important_headings_only:
                        mesh_heading_details = {
                            'Descriptor': descriptor_name,
                            'Descriptor UI': descriptor_ui,
                            'Is Major Topic': descriptor_major_topic
                        }
                        mesh_headings.append(mesh_heading_details)                    
                # Put the extracted PMID details into a tidy dictionary
                retrieved_data[pmid] = {
                    'Title': title,
                    'Abstract': abstract,
                    'Mesh Headings': mesh_headings
                }
                # print(f"PMID: {pmid}\nTitle: {title}\nAbstract: {abstract}\nMeSH Headings: {', '.join(mesh_headings)}\n{'-'*80}\n")
            return retrieved_data
        except:
            return None
    else:
        print(f"Error fetching articles: {efetch_response.status_code}")
        return None


def get_retrieved_article_details_from_query(query, articles_to_retrieve=10, remove_stop_words = False):
    '''Wrapper function that takes in a query and output article details.'''
    # Get response from entrez first via esearch
    query_response = get_query_response(query, remove_stop_words=remove_stop_words, articles_to_retrieve=articles_to_retrieve)
    translated_query = query_response['esearchresult']['querytranslation']
    retrieved_ids = query_response['esearchresult']['idlist']
    # Extract all the details of esearch's retrieved articles using efetch
    retrieved_article_details = get_article_details_from_id(retrieved_ids)
    return retrieved_article_details, translated_query

def create_article_semantic_neighbourhood(heading_entries:list, article_entries:list):
    filtered_headings = [x['name'] for x in heading_entries]
    semantic_neighbourhoods = []
    for article_entry in article_entries:
        expansion_set = article_entry['headings'] # these are the important headings attached to the articles
        if len(expansion_set)!=0:
            article_semantic_neighbourhood_chunks = []
            first_term_inserted = False
            for i, val in enumerate(expansion_set):
                # First, vet out article mesh headings that are not in the filtered heading entries being passed in
                if val not in filtered_headings:
                    continue
                if first_term_inserted: # first term of set
                    article_semantic_neighbourhood_chunks.append(f'AND "{val}"[MeSH Terms]')
                    first_term_inserted = True # this code added here just for safety, logically not necessary
                # elif i == (len(expansion_set)-1): # final term of set
                #     chunk_fragments_to_expand.append(f'AND "{val}"[MeSH Terms])')
                else:
                    article_semantic_neighbourhood_chunks.append(f'("{val}"[MeSH Terms]')
                    first_term_inserted = True
                    # chunk_fragments_to_expand.append(f'AND "{val}"[MeSH Terms]')
            if len(article_semantic_neighbourhood_chunks)!=0: # 0 will happen if all the fragments are not relevant semantically
                article_semantic_neighbourhood_chunks.append(')')
                article_semantic_neighbourhood = ' '.join(article_semantic_neighbourhood_chunks)
                semantic_neighbourhoods.append(article_semantic_neighbourhood)
    return semantic_neighbourhoods


def create_semantic_neighbourhood_query(heading_entries:list, article_entries:list):
    '''
    Takes in one list of filtered heading and article entries each and creates a Boolean requery term out of it.
    '''    
    semantic_neighbourhoods = create_article_semantic_neighbourhood(heading_entries, article_entries)
    if semantic_neighbourhoods != []:
        query_with_semantic_neighbourhoods = " OR ".join(semantic_neighbourhoods)
    else:
        query_with_semantic_neighbourhoods = ''
    return query_with_semantic_neighbourhoods

