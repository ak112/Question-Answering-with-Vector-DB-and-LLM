# Question Answering with Vector Database and Large Language Model

This repository contains an implementation of a question-answering system using a vector database and a large language model (LLM). The system is designed to retrieve relevant images based on natural language queries and generate textual descriptions of the entities mentioned in the queries.

## Problem Statement

The objective of this assignment is to showcase the ability to solve problems and the methodology for tackling specific tasks within the designated scope. The assignment is divided into two main sections:

1. Constructing an image vectors database using the provided data and developing a search feature that accepts natural language text as input and retrieves relevant images through vector search capabilities.

2. Utilizing an open-source Large Language Model (LLM) to process natural language queries as input and return the output textual descriptions of the entities mentioned within those queries.

## Approach

The assignment is implemented using Python and various libraries to achieve the desired functionality. The key steps involved in the implementation are as follows:

1. **Data Preparation**: The dataset used for this assignment is the Pascal VOC 2012 dataset, which contains images and their corresponding annotations. The dataset is loaded using the Hugging Face `datasets` library.

2. **Vector Database Creation**: The images from the dataset are preprocessed and converted to RGB format. The CLIP (Contrastive Language-Image Pre-training) model is used to extract image features. The extracted features are then stored in a FAISS (Facebook AI Similarity Search) index to create a vector database for efficient image retrieval.

3. **Search Functionality**: A search function is implemented that takes a natural language query as input, processes it using the CLIP model to extract text features, and performs similarity search using FAISS to retrieve the most relevant image from the vector database.

4. **LLM for Entity Description**: An open-source LLM, specifically the Mistral-7B model from Unsloth, is utilized to generate textual descriptions of the entities mentioned in the input query. The model is loaded using the Unsloth library, and a custom prompt is used to guide the model in generating concise descriptions for each entity.

## Libraries Used

The following libraries are used in this assignment:

- `datasets`: Loading the Pascal VOC 2012 dataset.
- `transformers`: Loading pre-trained models (CLIP, Mistral-7B) and tokenizers.
- `torch` and `torchvision`: PyTorch libraries for tensor operations and image processing.
- `faiss-cpu`: FAISS library for efficient similarity search.
- `PIL`: Python Imaging Library for image manipulation.
- `numpy`: Numerical computing library for array operations.
- `gptq`: GPT-Q library for model quantization.
- `accelerate`: Library for optimizing and accelerating model training and inference.
- `bitsandbytes`: Library for optimizing model quantization.
- `peft`: Library for parameter-efficient fine-tuning of language models.
- `unsloth`: Library for fast and memory-efficient language model inference.

## Advantages of the Libraries

The libraries used in this assignment offer several advantages:

- `transformers` and `unsloth` provide easy access to state-of-the-art pre-trained models like CLIP and Mistral-7B, enabling powerful image feature extraction and language generation capabilities.
- `faiss-cpu` enables efficient similarity search in high-dimensional vector spaces, allowing fast retrieval of relevant images.
- `gptq`, `accelerate`, `bitsandbytes`, and `peft` optimize model quantization, memory usage, and performance, enabling the use of large models on resource-constrained systems.
- `unsloth` provides fast and memory-efficient inference for language models, enabling real-time generation of entity descriptions.

## Constraints

The assignment had the following constraints:

- The use of external vector database APIs, such as Pinecone, was not permitted. The vector database had to be maintained locally.
- The use of search APIs for performing search operations was not allowed. The search functionality had to be implemented from scratch.
- The chosen LLM had to have a minimum of 7 billion parameters and a GPU vRAM consumption not exceeding 16GB.

## Conclusion

This assignment demonstrates the implementation of a question-answering system that combines vector search capabilities with a large language model. By leveraging the power of pre-trained models and efficient libraries, the system is able to retrieve relevant images based on natural language queries and generate concise descriptions of the entities mentioned in the queries.

The use of libraries like Unsloth and FAISS enables fast and memory-efficient inference, making it possible to work with large models on resource-constrained systems. The assignment showcases the potential of combining visual and textual information for enhanced question-answering capabilities.

For more details on the implementation and usage, please refer to the provided Colab notebook.