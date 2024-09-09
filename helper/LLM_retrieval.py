# ---------------------
# IMPORT ALL LIBRARIES
# ---------------------

# Extra components required when dealing with imported files
from haystack.components.converters import MarkdownToDocument, PyPDFToDocument, TextFileToDocument
from haystack.components.preprocessors import DocumentSplitter, DocumentCleaner
from haystack.components.routers import FileTypeRouter
from haystack.components.joiners import DocumentJoiner

# Rest is as per normal
from haystack.utils import Secret
from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.writers import DocumentWriter
from haystack.document_stores.types import DuplicatePolicy
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.embedders import HuggingFaceAPITextEmbedder, HuggingFaceAPIDocumentEmbedder
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.embedders import OpenAITextEmbedder
# from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder, OllamaTextEmbedder


# Initialize retrievers and readers
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators import OpenAIGenerator
# from haystack_integrations.components.generators.ollama import OllamaGenerator, OllamaChatGenerator
from haystack.components.readers import ExtractiveReader

import os


# --------------
# MANAGER CLASS
# --------------
class RAGManager():
    ''' 
    An overaching wrapper class that 1) manages multiple document stores 2) the retrieval and addition operations done to these stores as well as 3) their chat histories.\n
    '''
    
    def __init__(self, api_key, need_generation = True, need_reader = True, need_file_router=False):
        self.api_key = api_key
        self.document_store = InMemoryDocumentStore(embedding_similarity_function='cosine') 
        self.preprocessing_pipeline = initialize_document_preprocessing_pipeline(self.document_store, self.api_key, split_length = 250, split_overlap = 75, need_file_router=need_file_router)
        self.retrieval_pipeline = initialize_retrieval_pipeline(self.document_store, self.api_key, chosen_prompt = 'Model 1', need_generation=need_generation, need_reader=need_reader)
        self.chat_histories = [[],[]]
        self.need_generation = need_generation
        self.need_reader = need_reader  
    
    # Reads individual documents from streamlit file uploader which is already converted into python string
    # Tag options already handled externally before passing into the function (Document filter)
    def read_individual_documents(self, doc_store_docs):
        populate_document_store_from_fileUploader(self.document_store, doc_store_docs, self.preprocessing_pipeline)

    # Can add error handling here in case user does not pass in tag map (defaults to first tag option). Meant for read-directory branch.
    # Handles tag options within the function since creation of 'Document' objects is within the function
    def read_documents(self, document_directory, tag_map):
        populate_document_store_from_directory(self.preprocessing_pipeline, document_directory, tag_map)
    
    def answer_user(self, user_question, chosen_tag = 0, threshold = 0.5, filter = None):
        try:
            response = answer_user_query(user_question, self.retrieval_pipeline, threshold, need_generation=self.need_generation, need_reader=self.need_reader, filter=filter)
        # For when document store does not have any items or an error occured.
        except Exception as e:
            print('Error at rag manager answer_user stage:', e)
            response = 'I have not been trained on any documents yet. Please upload some documents to build up my knowledge.'
        self.chat_histories[chosen_tag].append(f"You: {user_question}")
        self.chat_histories[chosen_tag].append(f"Assistant: {response}")

    def clear_chat(self):        
        self.chat_histories = [[],[]]

    def change_retrieval_prompt(self, model_option):
        self.retrieval_pipeline = initialize_retrieval_pipeline(self.document_store, self.api_key, chosen_prompt = model_option, need_generation=self.need_generation, need_reader=self.need_reader)
        
        
class RetrievalManager():
    ''' 
    An overaching wrapper class that manages a single document store and the retrieval and addition operations done to said store.\n
    '''

    def __init__(self, api_key, need_file_router=False, embedding_model='openai'):
        self.api_key = api_key
        self.document_store = InMemoryDocumentStore(embedding_similarity_function='cosine') 
        self.preprocessing_pipeline = initialize_document_preprocessing_pipeline(self.document_store, self.api_key, split_length = 400, split_overlap = 50, 
                                                                                 need_file_router=need_file_router, embedding_model=embedding_model)
        self.retrieval_pipeline = initialize_retrieval_pipeline(self.document_store, self.api_key, chosen_prompt = 'Model 1', need_generation=False, 
                                                                need_reader=False, max_retrieved_docs=40, embedding_model=embedding_model)   

    def read_individual_documents(self, doc_store_docs):
        # populate_document_store_from_fileUploader(self.document_store, doc_store_docs, self.api_key)
        populate_document_store_from_fileUploader(self.document_store, doc_store_docs, self.preprocessing_pipeline)

    def answer_user(self, user_question, threshold = 0.1, filter = None):
        return answer_user_query(user_question, self.retrieval_pipeline, threshold, need_generation= False, need_reader=False, filter = filter)
    

# ---------------------------
# DEFINE AUXILLIARY FUNCTIONS
# ---------------------------

# Auxilliary function to get location of a single answer
def get_answer_location(answer_object, threshold, need_reader):
    most_accurate_score = 0
    # selected_answer_index = 0
    answer_location = ''
    # Evaluation based on exterior meta accuracy tag (the reader score)
    if need_reader:
        reader_answers = answer_object['reader']['answers']
        # for idx, extracted_answer in enumerate(reader_answers):
        for extracted_answer in reader_answers:
            if extracted_answer.to_dict()['init_parameters']['score'] > most_accurate_score:
                most_accurate_score = extracted_answer.to_dict()['init_parameters']['score']
                # selected_answer_index = idx                    
                answer_location = extracted_answer.to_dict()['init_parameters']['document']['meta']['file_path'].split('\\')[-1]      
    else:
        retriever_answers = answer_object['embedding_retriever']['documents']
        for item in retriever_answers:  
            # temp_dict = item
            score = item.to_dict()['score']
            if score > most_accurate_score:
                most_accurate_score = score
                answer_location = item.to_dict()['file_path'].split('\\')[-1]
    print('Most accurate score is:', most_accurate_score, 'with threshold:', threshold)
    print('Document location is', answer_location)
    # For the case where the most accurate score is still shit (below 50%)
    if most_accurate_score < threshold:        
        # selected_answer_index = None    
        answer_location = None    
        print('Cannot find from any document')
    return answer_location, most_accurate_score

# Auxilliary function to return a list of top x likely candidates based on user request
def select_top_candidates(answer_object, need_reader):
    # This part handles retriever candidates
    retriever_selected_candidates = {}
    reader_selected_candidates = {}
    retrieved_documents = answer_object['embedding_retriever']['documents']
    for item in retrieved_documents:  
        temp_dict = item.to_dict()
        name = temp_dict['file_path'].split('\\')[-1]
        score = temp_dict['score']
        if name not in retriever_selected_candidates: # if new candidate
            retriever_selected_candidates[name] = score
        elif retriever_selected_candidates[name] < score: # if candidate appeared but score lower
            retriever_selected_candidates[name] = score
    # This part handles reader candidates    
    # TAKE NOTICE: this part contains commented code for for an alternative extraction of retriever score
    if need_reader:
        reader_documents = answer_object['reader']['answers']
        for item in reader_documents: 
            temp_dict = item.to_dict()['init_parameters'] 
            try: 
                name = temp_dict['document']['meta']['file_path'].split('\\')[-1]
                # score = temp_dict['document']['score']
                second_score = temp_dict['score']
                if name not in reader_selected_candidates: # if new candidate
                    # retriever_selected_candidates[name] = first_score
                    reader_selected_candidates[name] = second_score
                # if retriever_selected_candidates[name] < first_score: # if candidate appeared but score lower
                #     print('Trigger first')
                #     retriever_selected_candidates[name] = first_score
                elif reader_selected_candidates[name] < second_score:
                    print('Trigger second')
                    reader_selected_candidates[name] = second_score
            except:
                print('Empty answer')
                continue
    sorted_candidates = {
        'retriever_selected_candidates' : dict(sorted(retriever_selected_candidates.items(), key=lambda item: item[1], reverse=True)),
        'reader_selected_candidates' : dict(sorted(reader_selected_candidates.items(), key=lambda item: item[1], reverse=True))
    }
    return sorted_candidates


# ------------------------
# POPULATE VECTOR DATABASE
# ------------------------

def initialize_document_preprocessing_pipeline(document_store, api_key, split_length, split_overlap, need_file_router, embedding_model):
    print('Initializing preprocessing pipeline.')
    openai_key = api_key

    # --------Initialize classes
    if need_file_router:
        file_type_router = FileTypeRouter(mime_types=["text/plain", "application/pdf", "text/markdown"])
        text_file_converter = TextFileToDocument()
        markdown_converter = MarkdownToDocument()
        pdf_converter = PyPDFToDocument()
        document_joiner = DocumentJoiner()  
    document_cleaner = DocumentCleaner()
    document_splitter = DocumentSplitter(split_by="word", split_length=split_length, split_overlap=split_overlap)
    if embedding_model == 'openai':
        document_embedder = OpenAIDocumentEmbedder(Secret.from_token(openai_key), model="text-embedding-3-large", meta_fields_to_embed=['file_path'])
    else:
        document_embedder =  SentenceTransformersDocumentEmbedder() # default model using all-mpnet-base-v2
    # document_embedder =  SentenceTransformersDocumentEmbedder(model='sentence-transformers/sentence-t5-xxl') # default model using all-mpnet-base-v2    
    # document_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api", api_params={"model": "w601sxs/b1ade-embed"}, token=Secret.from_token('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'))    #
    # document_embedder = OpenAIDocumentEmbedder(Secret.from_token(openai_key), model="text-embedding-3-large", meta_fields_to_embed=['file_path'])
    # document_embedder = OllamaDocumentEmbedder(model="mxbai-embed-large", url="http://localhost:11434/api/embeddings") 
    document_writer = DocumentWriter(document_store = document_store, policy = DuplicatePolicy.SKIP)  # different syntax compared to when you are using a dataframe

    # -------Create pipeline and components
    pipeline = Pipeline()
    if need_file_router:
        pipeline.add_component(instance=file_type_router, name="file_type_router")
        pipeline.add_component(instance=text_file_converter, name="text_file_converter")
        pipeline.add_component(instance=markdown_converter, name="markdown_converter")
        pipeline.add_component(instance=pdf_converter, name="pypdf_converter")
        pipeline.add_component(instance=document_joiner, name="document_joiner") # merge all input types into the same list    
    pipeline.add_component(instance=document_cleaner, name="document_cleaner")
    pipeline.add_component(instance=document_splitter, name="document_splitter")
    pipeline.add_component(instance=document_embedder, name="document_embedder")
    pipeline.add_component(instance=document_writer, name="document_writer")

    # -------Connect components
    if need_file_router:
        pipeline.connect("file_type_router.text/plain", "text_file_converter.sources")
        pipeline.connect("file_type_router.application/pdf", "pypdf_converter.sources")
        pipeline.connect("file_type_router.text/markdown", "markdown_converter.sources")
        pipeline.connect("text_file_converter", "document_joiner")
        pipeline.connect("pypdf_converter", "document_joiner")
        pipeline.connect("markdown_converter", "document_joiner")
        # Consider adding a custom component here (or after document joiner) that adds metadata
        pipeline.connect("document_joiner", "document_cleaner") 
    # Can add here too since in and output to doc cleaner are documents - verify what docs are fed before adding metadata
    # Check https://docs.haystack.deepset.ai/reference/data-classes-api but seems like this thing doesn't work
    pipeline.connect("document_cleaner", "document_splitter")
    pipeline.connect("document_splitter", "document_embedder")
    pipeline.connect("document_embedder", "document_writer")

    print('Preprocessing pipeline initialized.')
    return pipeline


# def populate_document_store_from_fileUploader(document_store, document_list, api_key):
def populate_document_store_from_fileUploader(document_store, document_list, preprocessing_pipeline):
    print('Running preprocessing pipeline.')
    print('Pre image-loading document store population:', document_store.count_documents())
    # openai_key = api_key
    # document_embedder = OpenAIDocumentEmbedder(Secret.from_token(openai_key), model="text-embedding-3-large", meta_fields_to_embed=["title"])
    # # document_embedder = OllamaDocumentEmbedder(model="mxbai-embed-large", url="http://localhost:11434/api/embeddings") 
    # docs_with_embeddings = document_embedder.run(document_list)
    # document_store.write_documents(docs_with_embeddings["documents"])    
    preprocessing_pipeline.run({
        'document_cleaner': {"documents": document_list}
    })
    print('Post image-loading document store population:', document_store.count_documents())
    # return pipeline


# def populate_document_store(file_path, document_store, pipeline):
# def populate_document_store_from_directory(first_preprocessing_pipeline, second_preprocessing_pipeline, file_path, tag_options, tag_map):
def populate_document_store_from_directory(first_preprocessing_pipeline, file_path, tag_map):

    print('File path passed in:', file_path)
    # Execute pipeline    
    print('Running preprocessing pipeline.')
    list_of_files = [os.path.join(file_path, f) for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    print('The files to move to vector database are: '+str(list_of_files))
    # Run pipeline one by one on the files as we want to add tags
    # Very tedious, in future, to get rid of pipeline altogether
    for files in list_of_files:
        file_name = files.split('\\')[-1]
        # if tag_map[file_name]==tag_options[1]: # If the file belongs to the second tag
        #     # Run second preprocessing pipeline on one specific file
        #     second_preprocessing_pipeline.run({
        #         "file_type_router": {                                     
        #             "sources": [files]
        #         },          
        #         "pypdf_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}},
        #         "text_file_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}},
        #         "markdown_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}}
        #     })
        # else:
        #     # Run first preprocessing pipeline on one specific file
        first_preprocessing_pipeline.run({
            "file_type_router": {                                     
                "sources": [files]
            },          
            "pypdf_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}},
            "text_file_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}},
            "markdown_converter": {"meta": {"tag": tag_map[file_name], 'file_type':'Document'}}
        })
    
    # print('Final document store population:', document_store.count_documents())
    # print('Final second document store population:', second_document_store.count_documents())
    # return document_store, second_document_store


# ------------------------
# ANSWER USER QUERY
# ------------------------

def initialize_retrieval_pipeline(document_store, api_key, chosen_prompt, need_generation, need_reader, embedding_model, max_retrieved_docs = 10):    
    # Open AI key
    openai_key = api_key
    # Define prompts and engage the one chosen by user.
    templates = {
        'Model 1': """
                Answer the question based on the given context in either English or Malay, depending on the language of the question.
                If the answers are not found in the context, reply "No answers found in the given documents." if you are replying in English or "Tiada jawapan dalam dokumen yang diberi." if you are replying in Malay.

                Context:
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}

                Question: {{ question }}
                Answer:
                """,
        'Model 2': """
                Answer or give details to the User Input based on the given context in either English or Malay, depending on the language of the User Input.
                If you are not able to answer or give details based on the context, reply "No answers found in the given documents." if you are replying in English or "Tiada jawapan dalam dokumen yang diberi." if you are replying in Malay.

                Context:
                {% for document in documents %}
                    {{ document.content }}
                {% endfor %}

                User Input: {{ question }}
                Answer:
                """
    } 

    template = templates[chosen_prompt]
    print('Retrieval pipeline initialized with prompt:', chosen_prompt)

    # --------Initialize classes
    # Creating question and answer pipeline
    print('Initializing Q&A pipeline.')
    if embedding_model == 'openai':
        text_embedder = OpenAITextEmbedder(api_key=Secret.from_token(openai_key), model="text-embedding-3-large")
    else:
        text_embedder = SentenceTransformersTextEmbedder()
    # text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api", api_params={"model": "w601sxs/b1ade-embed"}, token=Secret.from_token('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'))

    # --------Add components
    pipe = Pipeline()    
    pipe.add_component("text_embedder", text_embedder) 
    # pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model='sentence-transformers/sentence-t5-xxl')) 
    # pipe.add_component("text_embedder", OllamaTextEmbedder(model="mxbai-embed-large", url="http://localhost:11434/api/embeddings"))
    pipe.add_component("embedding_retriever", InMemoryEmbeddingRetriever(document_store=document_store, top_k=max_retrieved_docs)) # the documents in document store has already undergone the pipeline above
    pipe.add_component("prompt_builder", PromptBuilder(template=template))
    if need_reader:
        reader = ExtractiveReader()
        reader.warm_up()
        pipe.add_component('reader', reader)
    if need_generation:
        pipe.add_component("llm", OpenAIGenerator(model="gpt-4o", api_key=Secret.from_token(openai_key)))
        # pipe.add_component("llm", OllamaGenerator(model="mistral",
        #                                             url = "http://localhost:11434/api/generate",
        #                                             generation_kwargs={
        #                                             "num_predict": 100,
        #                                             "temperature": 0.9,
        #                                             }))
    # Connect everything together
    pipe.connect("text_embedder.embedding", "embedding_retriever.query_embedding")
    pipe.connect("embedding_retriever", "prompt_builder.documents")
    if need_reader:
        pipe.connect("embedding_retriever.documents", "reader.documents")
    if need_generation:
        pipe.connect("prompt_builder", "llm")
    print('Q&A pipeline initialized.')
    return pipe


def answer_user_query(query, pipe, location_threshold, need_generation, need_reader, filter, max_qa_docs = 2):      
    # Truncate the query (necessary if using api embedder)
    words = query.split()    
    truncated_words = words[:4000]    
    query = ' '.join(truncated_words)

    print('Running Q&A pipeline on query:', query)
    # Initialize pipeline arguments
    data = {            
            "text_embedder": {"text": query},
            "embedding_retriever": {
                # "query": query,
                "filters": filter,
  		      },
            "prompt_builder": {"question": query},
        }
    if need_generation:
        data["llm"] = {"generation_kwargs": {"max_tokens": 800}} # openai
        # data["llm"] = {"generation_kwargs": {"max_new_tokens": 800}} # mistral
    if need_reader:
        data["reader"] =  {"query": query, "top_k": max_qa_docs}
    # Run the pipeline
    answer_object = pipe.run(
        data = data,        
        include_outputs_from = {"embedding_retriever"} if need_reader==False else {"embedding_retriever", "reader"}         
    )
    # For RAG systems
    if need_generation:
        actual_answer = answer_object["llm"]["replies"][0]
        try:
            document_with_answer, most_accurate_score = get_answer_location(answer_object, threshold=location_threshold, need_reader=need_reader)
        except Exception as e:
            print(e)
            document_with_answer = 'Not Applicable'        
        # try:
        #     document_page_1 = reader_answers[selected_answer_index].to_dict()['init_parameters']['meta']['answer_page_number']
        # except:
        #     document_page_1 = ''
        # try:
        #     document_page_2 = reader_answers[selected_answer_index].to_dict()['init_parameters']['document']['meta']['page_number']
        # except:
        #     document_page_2 = ''
        if ('tiada jawapan' in actual_answer.lower()) or ('no answers' in actual_answer.lower()):
            document_with_answer = 'Not Applicable'    # often generation conflicts with extractive reader, so we are forced to reckon with this
        try:
            response = f"Case Assistant: {actual_answer} **This information is retrieved from document:** '{document_with_answer}'."
        except:
            response = f"Case Assistant: {actual_answer}"
        return response
    # For simple retrieval operations
    else:
        top_k_candidates = select_top_candidates(answer_object, need_reader=need_reader)
        return top_k_candidates




