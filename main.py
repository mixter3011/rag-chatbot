import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as VectorStorePinecone
from langchain.llms import HuggingFaceHub
from pinecone import Pinecone, ServerlessSpec
from langchain import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter

class ChatBot():
    def __init__(self):
        
        load_dotenv()

        
        loader = TextLoader('/Users/sen/Desktop/vscode/python/rag-chatbot/data.txt')
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        
        embeddings = HuggingFaceEmbeddings()

        
        pc = Pinecone(
            api_key=os.getenv('PINECONE_API_KEY')  
        )

        index_name = "rag-demo1"

        
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

        
        retriever = docsearch.as_retriever(
            search_type="mmr",  
            search_kwargs={"k": 5}  
        )

        
        repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        llm = HuggingFaceHub(
            repo_id=repo_id,
            model_kwargs={"temperature": 0.5, "top_p": 0.7, "top_k": 40},
            huggingfacehub_api_token=os.getenv('HUGGINGFACE_ACCESS_TOKEN')
        )

        
        template = """You are a helpful AI assistant for students of MRIIRS (Manav Rachna International Institute of Research and Studies). 
        Provide a precise and informative answer based strictly on the given context. 
        If the context does not contain sufficient information to answer the question, 
        clearly state that you cannot find the specific information.

        Context:
        {context}

        Question: {question}

        Helpful Answer:"""

        
        self.prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        
        self.rag_chain = (
            {
                "context": lambda x: self._format_docs(retriever.get_relevant_documents(x)),
                "question": RunnablePassthrough()
            } 
            | self.prompt 
            | llm
        )

    def _format_docs(self, docs):
        """
        Format retrieved documents into a readable context string.
        Helps ensure the context is clear and readable for the LLM.
        """
        print("Retrieved Documents:")
        for i, doc in enumerate(docs, 1):
            print(f"Document {i}:")
            print(f"Content: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}\n")
        
        return "\n\n".join(doc.page_content for doc in docs)

    def ask(self, question):
        """
        Ask a question and get an answer based on the context.
        Added error handling and debug information.
        """
        try:
            print(f"Processing question: {question}")
            
            
            output = self.rag_chain.invoke(question)
            
            
            answer = output.split("Helpful Answer:")[-1].strip()
            
            print(f"Generated Answer: {answer}")
            return answer
        
        except Exception as e:
            print(f"Error processing question: {e}")
            return "I apologize, but I'm having trouble processing your question. Could you please rephrase it?"
