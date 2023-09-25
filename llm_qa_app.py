import os
import time
from dotenv import load_dotenv, find_dotenv
import pinecone
import tiktoken

# Constants
DIMENSION_OF_EMBEDDINGS = 1536

# Load environment variables
load_dotenv(find_dotenv(), override=True)
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENV = os.environ.get('PINECONE_ENV')

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Load a document based on its format
def load_document(file):
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data

# Load text from Wikipedia
def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data

# Chunk text data
def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# Calculate embedding cost
def print_embedding_cost(texts):
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')

# Insert or fetch embeddings in Pinecone
def insert_or_fetch_embeddings(index_name):
    from langchain.vectorstores import Pinecone
    from langchain.embeddings.openai import OpenAIEmbeddings

    embeddings = OpenAIEmbeddings()

    if index_name in pinecone.list_indexes():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings ...', end='')
        pinecone.create_index(index_name, dimension=DIMENSION_OF_EMBEDDINGS, metric='cosine')
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')

    return vector_store

# Delete a Pinecone index
def delete_pinecone_index(index_name='all'):
    if index_name == 'all':
        indexes = pinecone.list_indexes()
        print('Deleting all indexes ... ')
        for index in indexes:
            pinecone.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name} ...', end='')
        pinecone.delete_index(index_name)
        print('Ok')

# Ask a question and get an answer
def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = chain.run(q)
    return answer

# Ask a question with memory
def ask_with_memory(vector_store, question, chat_history=[]):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

# Main function
def main():
    data = load_document('files/planting.pdf')

    if data is None:
        return

    print(f'You have {len(data)} pages in your data')
    print(f'There are {len(data[20].page_content)} characters in the page')

    chunks = chunk_data(data)
    print(len(chunks))

    print_embedding_cost(chunks)
    delete_pinecone_index()

    index_name = 'askadocument'
    vector_store = insert_or_fetch_embeddings(index_name)

    q = 'What is the whole document about?'
    answer = ask_and_get_answer(vector_store, q)
    print(answer)

    i = 1
    print('Write "Quit" or "Exit" to quit.')
    while True:
        q = input(f'Question #{i}: ')
        i += 1
        if q.lower() in ['quit', 'exit']:
            print('Quitting ... bye bye!')
            time.sleep(2)
            break

        answer = ask_and_get_answer(vector_store, q)
        print(f'\nAnswer: {answer}')
        print(f'\n {"-" * 50} \n')

if __name__ == "__main__":
    main()


# QUERY FROM WIKIPEDIA OR SIMILAR ONLINE SOURCES:

# delete_pinecone_index()
#
# data = load_from_wikipedia('ChatGPT', 'tr')
# chunks = chunk_data(data)
# index_name = 'Mercedes-Benz'
# vector_store = insert_or_fetch_embeddings(index_name)


# Asking With Memory:

# chat_history = []
# question = 'How many stages does the planting design consist of?'
# result, chat_history = ask_with_memory(vector_store, question, chat_history)
# print(result['answer'])
# print(chat_history)
#
# question = 'What are they?'
# result, chat_history = ask_with_memory(vector_store, question, chat_history)
# print(result['answer'])
# print(chat_history)
