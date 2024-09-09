import streamlit as st
import helper.pubmed_search as pubs
from helper.test import mocked_heading_entries, mocked_article_entries
from helper.pubmed_search import QueryExpansionManager
from helper.config import api_key
import pandas as pd
import json

# Generate default query expansion manager
if 'current_model' not in st.session_state:
    st.session_state.current_model = "w601sxs/b1ade-embed"
if 'query_retrieval_manager' not in st.session_state:
    st.session_state.query_expansion_manager=QueryExpansionManager(st.session_state.current_model, 'helper/descriptors.json')
if 'true_false_mapper' not in st.session_state:
    st.session_state.true_false_mapper= {'True': True, 'False': False}
if 'heading_entries' not in st.session_state:
    st.session_state.heading_entries = []
if 'article_entries' not in st.session_state:
    st.session_state.article_entries = []
if 'heading_entries_df' not in st.session_state:
    st.session_state.heading_entries_df = []
if 'article_entries_df' not in st.session_state:
    st.session_state.article_entries_df = []
if 'semantic_neighbourhood_queries' not in st.session_state:
    st.session_state.semantic_neighbourhood_queries = ''
if 'requeried_article_entries' not in st.session_state:
    st.session_state.requeried_article_entries = ''
if 'final_article_entries' not in st.session_state:
    st.session_state.final_article_entries = ''

# --------------------------
# INITIALIZE GLOBAL VARIABLES
# --------------------------
st.set_page_config(layout="wide", page_title='Medical Query Assistant')


# ------------
#  COMPONENTS
# ------------
@st.fragment
def display_search_results():
    st.header(":blue[Initial Search Result]")
    with st.container(border=True):
        st.subheader('Relevant Headings')
        with st.expander('Display Headings', expanded=False):
            st.table(st.session_state.heading_entries_df)
        st.subheader('Relevant Articles')
        with st.expander('Display Articles', expanded=False):
            st.table(st.session_state.article_entries_df)

@st.fragment
def requery_interface():
    st.header(':green[Expand Your Query with Suggested Terms]')
    # st.session_state.requery_interface_reruns += 1
    tab1, tab2 = st.tabs(['Suggested Expansion', 'Self Key In'])
    with tab1:
        with st.spinner('Generating query suggestion terms...'):
            semantic_neighbourhoods = pubs.create_article_semantic_neighbourhood(st.session_state.heading_entries, st.session_state.article_entries)
            st.session_state.semantic_neighbourhood_queries = pubs.create_semantic_neighbourhood_query(st.session_state.heading_entries, 
                                                                                                       st.session_state.article_entries)   
        with st.form('submit_semantic_cluster'):     
            st.subheader('Semantic Clusters (Suggested Expansion Term)')
            st.write('The current query is: ')
            st.write(st.session_state.semantic_neighbourhood_queries)
            if st.form_submit_button('Confirm re-submitted query.', type='primary'):
                with st.spinner('Extracting requery results...'):
                    _, st.session_state.requeried_article_entries = st.session_state.query_expansion_manager.get_entries_from_query(st.session_state.semantic_neighbourhood_queries, requery = True, 
                                                                                                                    articles_to_retrieve= 20, remove_stop_words= False)
                st.session_state.requeried_article_entries_df = pd.DataFrame(st.session_state.requeried_article_entries).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)    
                st.write('**The final suggested articles are**:') 
                st.session_state.final_article_entries = pd.concat([st.session_state.requeried_article_entries_df, st.session_state.article_entries_df]).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)   
                st.table(st.session_state.final_article_entries.head(10))                          
        # with st.form('expanded query'):
        #     total_queries = ''
        #     semantic_neighbourhood_queries = {}
        #     for idx, item in enumerate(semantic_neighbourhoods):
        #         semantic_neighbourhood_queries[str(idx)] = st.checkbox(str(item), key=str(item))
        #     # for key, item in semantic_neighbourhood_queries.items():
        #         if semantic_neighbourhood_queries[str(idx)]:
        #             total_queries += str(item)
        #     st.write(total_queries)
        #     st.write('Test only')
        #     model_submmitted = st.form_submit_button("Confirm Desired Model", type='primary')
        #         # if model_submmitted:          
    with tab2:
        heading_threshold = 50
        st.subheader('Build Query from Suggested Headings')
        # for idx, item in st.session_state.heading_entries_df.iterrows():
        #     if item['suitability'] > heading_threshold:
        #         # checkbox_value = False
        #         checkbox_label = f'{item['name']} (recommended)'                
        #     else:
        #         # checkbox_value = False
        #         checkbox_label = f'{item['name']}'
        #     st.write(checkbox_label)
        # manually_expanded_query = st.text_input('Key in your boolean or natural language query based on the expanded term.')
        # if manually_expanded_query:
        #     with st.spinner('Extracting requery results...'):
        #         _, requeried_article_entries = st.session_state.query_expansion_manager.get_entries_from_query(manually_expanded_query, requery = True, 
        #                                                                                                         articles_to_retrieve= 20, remove_stop_words= False)
        #         requeried_article_entries_df = pd.DataFrame(requeried_article_entries).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)  
        #     st.write('**The final suggested articles are**:') 
        #     st.session_state.final_article_entries = pd.concat([requeried_article_entries_df, st.session_state.article_entries_df]).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)   
        #     st.table(st.session_state.final_article_entries.head(10))     
    # with st.form():
    #     st.write('Examine your new query')

# ---------------------
# CREATE PAGE CONTENT
# ---------------------

st.title("Biomedical Query Assistant")
st.write('This assistant performs query suggestion and expansion on user queries to give more relevant returns to users on publicly available, ontological data.')

with st.sidebar:
    st.image('logos/US-NLM-PubMed-Logo.svg')
    with st.form('model_selection_form'):
        selected_model = st.selectbox('Choose your preferred embedding model:', ['w601sxs/b1ade-embed', 'dmis-lab/biobert-v1.1', 'openai'])
        if selected_model == 'openai':
            st.text_input('Please provide your openai key.')
        model_submmitted = st.form_submit_button("Confirm Desired Model", type='primary')
        if model_submmitted and (selected_model!=st.session_state.current_model):            
            st.session_state.query_expansion_manager=QueryExpansionManager(selected_model, 'helper/descriptors.json')
            st.session_state.current_model = selected_model        
        st.write(f'Current model used is: **{st.session_state.current_model}**')

# -- USER INPUT AND TRIGGER SEARCH --
with st.form('initial_search_form'):
    initial_search_query = st.text_input('Key in your natural language search here...')
    articles_to_return = st.slider('How many articles do you want to retrieve?', 1, 50, 10)
    removing_stop_words = st.selectbox('Do you wish to remove stopwords?', options=['False', 'True'])
    submitted = st.form_submit_button("Submit Query")
    if submitted:
        with st.spinner('Getting articles from PubMed...'):
            filtered_data = st.session_state.query_expansion_manager.get_entries_from_query(initial_search_query, requery=False, 
                                                                                     remove_stop_words=st.session_state.true_false_mapper[removing_stop_words])
            # I defined
            st.session_state.heading_entries = filtered_data[0]
            st.session_state.heading_entries_df = pd.DataFrame(st.session_state.heading_entries).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)
            st.session_state.article_entries = filtered_data[1]
            st.session_state.article_entries_df = pd.DataFrame(st.session_state.article_entries).drop_duplicates(['name']).sort_values(by=['suitability'], ascending=False)
st.markdown("<br>", unsafe_allow_html=True)

if (st.session_state.heading_entries != []) or (st.session_state.article_entries !=[]):
    display_search_results()    
    st.markdown("<br>", unsafe_allow_html=True)
    requery_interface()

