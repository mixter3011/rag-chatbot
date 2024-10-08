from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as VectorStorePinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

class ChatBot():
    def __init__(self):
        load_dotenv()

        
        loader = TextLoader('/Users/sen/Desktop/vscode/python/rag-chatbot/data.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
        docs = text_splitter.split_documents(documents)

        
        embeddings = HuggingFaceEmbeddings()

        
        pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')  
        )

        index_name = "langchain-demo"

        
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=768,  
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"  
                )
            )

        
        docsearch = VectorStorePinecone.from_documents(docs, embeddings, index_name=index_name)

        
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
        )

        
        template = """
        You are a helpful assistant for students of MRIIRS (Manav Rachna International Institute of Research and Studies). 
        Students will ask you questions about their college life, including academic policies, internships, exams, clubs, and other activities. 
        Use the following context from the student handbook to provide accurate answers to their questions. 
        If you don't know the answer, just say you don't know.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        
        self.rag_chain = (
            {"context": docsearch.as_retriever(), "question": RunnablePassthrough()} 
            | self.prompt 
            | llm
        )

    def ask(self, question):
        output = self.rag_chain.invoke(question)        
        answer = output.split("Answer:")[-1].strip()  
        return answer
