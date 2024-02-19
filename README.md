# AI Search Labs

This is a lab about using Azure AI Search together with embeddings of text data through Azure OpenAI Service to perform hybrid search to obtain most relevant search results.

## Prerequisites

* You need an Azure subscription
* You should have [created an Azure AI Search service resource](https://docs.microsoft.com/en-us/azure/search/search-create-service-portal) and obtained the [endpoint and admin key](https://docs.microsoft.com/en-us/azure/search/search-security-api-keys)
* You should have [created an Azure OpenAI Service resource](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/create-resource?pivots=web-portal), deployed at least the `text-embedding-ada-002` embedding model, and obtained the endpoint and key.

## Lab Steps

* Rename the `env.sample` file to `.env` and fill in the values for the Azure AI Search and Azure OpenAI Service resources.
* Follow the instructions in the `ai_search_lab.ipynb` notebook to complete the lab.
