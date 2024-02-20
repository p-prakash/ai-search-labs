import os
import argparse
import json
import time
from typing import Optional
from dotenv import load_dotenv
import numpy as np
import openai
from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient


def get_product_data():
    # load Product Enrichment Response p1.json
    with open('Product Enrichment Response p1.json') as f:
        data = json.load(f)
        products = data['products']

    # products = [
    #         {
    #             "productid": "1",
    #             "taxonomy": "category1",
    #             "title": "Product 1",
    #             "description": "Description of Product 1"
    #         },
    #         {
    #             "productid": "2",
    #             "taxonomy": "category2",
    #             "title": "Product 2",
    #             "description": "Description of Product 2"
    #         }
    #     ]
    
    return products

def create_search_index(index_client: SearchIndexClient, index_name: str):

    from azure.search.documents.indexes.models import (
        SearchIndex,
        SearchField,
        SearchFieldDataType,
        SimpleField,
        SearchableField,
        VectorSearch,
        VectorSearchProfile,
        HnswAlgorithmConfiguration,
    )
    
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="image_url", type=SearchFieldDataType.String),
        SearchableField(
            name="title",
            type=SearchFieldDataType.String,
            sortable=True,
            filterable=True,
        ),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchableField(name="metaData", type=SearchFieldDataType.String),
        SearchField(
            name="titleVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        ),
        SearchField(
            name="descriptionVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-config",
        ),
        SearchableField(
            name="taxonomy",
            type=SearchFieldDataType.String,
            sortable=True,
            filterable=True,
            facetable=True,
        ),
    ]
    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="my-vector-config", algorithm_configuration_name="my-algorithms-config")],
        algorithms=[HnswAlgorithmConfiguration(name="my-algorithms-config")],
    )

    index_client.create_index(SearchIndex(
        name=index_name, 
        fields=fields, 
        vector_search=vector_search))

    print(f'created index {index_name}')

def before_retry_sleep():
    print(f"Rate limited on the OpenAI embeddings API, sleeping before retrying...")
    time.sleep(2)
    
def generate_embeddings(taxonomy: Optional[str] = None):
    # Initialize openai 
    load_dotenv()
    openai.api_type = "azure"
    openai.api_base = os.environ.get('AOAI_ENDPOINT')
    openai.api_key = os.environ.get('AOAI_KEY')
    openai.api_version = "2023-05-15"
    aoai_deployment = os.environ.get('AOAI_EMBEDDINGS_DEPLOYMENT')

    # Get product data and generate embeddings for title and description
    products = get_product_data()
    products_with_embeddings = []

    for product in products:
        print(f'Generating embeddings for product {product["id"]}...')
        while True:
            try:
                products_with_embeddings.append({
                    "id": str(product['id']),
                    "metaData": product['metaData'],
                    "image_url": product['image'][0].get('url') if product['image'] else None,
                    "taxonomy": product['taxonomies'][0]['name'] if product['taxonomies'] else None,
                    "title": product['title'],
                    "description": product['description'],
                    "titleVector": openai.Embedding.create(engine= aoai_deployment,input=product['title']).data[0].embedding,
                    "descriptionVector": openai.Embedding.create(engine= aoai_deployment,input=product['description']).data[0].embedding,
                })
            except openai.error.RateLimitError as e:
                before_retry_sleep()
                continue
            break

    # dump products with embeddings to json file
    with open('products_with_embeddings.json', 'w') as f:
        json.dump(products_with_embeddings, f)
        
    return products_with_embeddings


##################################################  
#  Main function
#  1. Create search index (if not exists)
#  2. Get product data and generate embeddings
#  3. Upload documents to the index
##################################################

load_dotenv()

# Initialize Azure search client
key = os.environ.get('AI_SEARCH_KEY')
endpoint = os.environ.get('AI_SEARCH_ENDPOINT')
index_name = os.environ.get('AI_SEARCH_INDEX_NAME')
index_client: SearchIndexClient = SearchIndexClient(
    endpoint=endpoint, credential = AzureKeyCredential(key))

# Check if the index exists, if not create it
try:
    index_client.get_index(name=index_name)
    print(f'Index {index_name} already exists')
except ResourceNotFoundError as ex:
    create_search_index(index_client, index_name)

# Initialize search client
search_client = SearchClient(endpoint, index_name, AzureKeyCredential(key))

# Upload documents to the index
print('Generating embeddings and uploading documents to the index...')
search_client.upload_documents(documents=generate_embeddings())

print('Done')