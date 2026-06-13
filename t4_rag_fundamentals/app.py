import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import SecretStr

from commons.constants import OPENAI_API_KEY


# Create system prompt with:
# - role: explains the role for LLM and what it should do
# - Structure of User message, consists of 2 blocks:
#   - `RAG CONTEXT`: information retrieved on the Retrieval step based on user request
#   - `USER QUESTION`: The user's actual question
# - Instructions:
#   - Model must use only information from conversation
#   - Strictly forbid to answer questions that are not in the conversation or not present in `RAG CONTEXT`
_SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.

## Structure of User message, consists of 2 blocks:
    `RAG CONTEXT`: information retrieved on the Retrieval step based on user request
    `USER QUESTION`: The user's actual question
## Instructions:
    - Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
    - Cite specific sources when using information from the context.
    - Answer ONLY based on conversation history and RAG context.
    - If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

_USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION:
{query}"""


class MicrowaveRAG:

    def __init__(self, embeddings: OpenAIEmbeddings, llm_client: ChatOpenAI):
        self.llm_client = llm_client
        self.embeddings = embeddings
        self.vectorstore = self._setup_vectorstore()

    def _setup_vectorstore(self) -> VectorStore:
        """
        Load existing FAISS index from disk or create a new one.
        Returns:
              VectorStore: Initialized FAISS vectorstore.
        """
        
        # - Print a startup message
        print("🔄 Initializing Microwave Manual RAG System...")
        # - Check if 'microwave_faiss_index' folder already exists
        # - If yes, load the index from disk using FAISS.load_local()
        # - If no, call _create_new_index() to build and save a fresh index
        # - Return the vectorstore
        if os.path.exists("t4_rag_fundamentals/microwave_faiss_index"):
            print("✅ Found existing FAISS index. Loading from disk...")
            vectorstore = FAISS.load_local("t4_rag_fundamentals/microwave_faiss_index", self.embeddings)
            print("✅ FAISS index loaded successfully.")
            
        else:
            print("⚠️ No existing FAISS index found. Creating a new one...")
            vectorstore = self._create_new_index()
            print("✅ New FAISS index created and saved successfully.")
        return vectorstore

    def _create_new_index(self) -> VectorStore:
        """
        Load the manual, split into chunks, embed, and save a new FAISS index.
        Returns:
              VectorStore: Newly created and saved FAISS vectorstore.
        """
        
        print("📖 Loading text document...")

        # - Load 'microwave_manual.txt' using TextLoader
        loader = TextLoader("t4_rag_fundamentals/microwave_manual.txt", encoding='utf-8')
        documents = loader.load()
        
        print("✂️ Splitting document into chunks...")

        # - Split documents into chunks using RecursiveCharacterTextSplitter
        #   (chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", "."])
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "."]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")

        # - Create a FAISS vectorstore from chunks and self.embeddings using FAISS.from_documents()
        print("🔍 Creating embeddings and FAISS index...")
        vectorstore = FAISS.from_documents(chunks, self.embeddings)

        print("🔍 vectorstore created with FAISS index...")

        # - Save the index locally using vectorstore.save_local("t4_rag_fundamentals/microwave_faiss_index")
        # - Return the vectorstore
        vectorstore.save_local("t4_rag_fundamentals/microwave_faiss_index")
        return vectorstore

    def retrieve_context(self, query: str, k: int = 4, score=0.3):
        """
        Retrieve the context for a given query.
        Args:
              query (str): The query to retrieve the context for.
              k (int): The number of relevant documents(chunks) to retrieve.
              score (float): The similarity score between documents and query. Range 0.0 to 1.0.
        """
        
        print(f"\n🔗 STEP 1: RETRIEVAL \n{'-' * 100}")
        # - Search the vectorstore using similarity_search_with_relevance_scores() with k and score_threshold parameters
        results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k, score_threshold=score)
        
        # - Iterate over results, collect each doc's page_content, and print its relevance score
        context_parts = []
        for doc, relevance in results:
            print(f"Relevance Score: {relevance}")
            context_parts.append(doc.page_content)

        print("=" * 100)

        # - Return all collected chunks joined with "\n\n" as a single context string
        return "\n\n".join(context_parts)

    def augment_prompt(self, query: str, context: str):
        """
        Inject retrieved context and user query into the prompt template.
        Args:
              query (str): The user's question.
              context (str): Retrieved context from the vectorstore.
        Returns:
              str: Formatted prompt ready for the LLM.
        """
        
        print(f"\n🔗 STEP 2: AUGMENTATION\n{'-' * 100}")

        # - Format _USER_PROMPT template substituting {context} and {query}
        augmented_prompt = _USER_PROMPT.format(context=context, query=query)

        # - Print the resulting augmented prompt
        print(f"Augmented Prompt:\n{augmented_prompt}\n{'=' * 100}")

        # - Return the formatted string
        return augmented_prompt

    def generate_answer(self, augmented_prompt: str):
        """
        Send the augmented prompt to the LLM and return its response.
        Args:
              augmented_prompt (str): The prompt with injected context and query.
        Returns:
              str: The LLM-generated answer.
        """
        print(f"\n🔗 STEP 3: GENERATION\n{'-' * 100}")
        # - Build a messages list: [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=augmented_prompt)]
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=augmented_prompt)
        ]

        # - Invoke self.llm_client.generate with a batch of messages
        response = self.llm_client.invoke(messages)

        # - Print the response content
        print(f"LLM Response:\n{response.content}\n{'=' * 100}")

        # - Return the response content string
        return response.content


def main(rag: MicrowaveRAG):
    # - Print a welcome message
    print("👋 Welcome to the Microwave Manual RAG System!")

    # - Run an infinite loop that reads user input with input()
     

    while True:
        query = input("\n❓ Please enter your question about the microwave (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
    # - For each question execute the 3-step RAG pipeline:
    #   - Step 1 (Retrieval):   call rag.retrieve_context() to fetch relevant chunks
    #   - Step 2 (Augmentation): call rag.augment_prompt() to build the prompt
    #   - Step 3 (Generation):  call rag.generate_answer() to get the LLM answer

        rag_context = rag.retrieve_context(query)
        augmented_prompt = rag.augment_prompt(query, rag_context)
        answer = rag.generate_answer(augmented_prompt)
        print(f"\n💡 Answer: {answer}")




# Start the application by calling main() and passing a MicrowaveRAG instance:

main(
    MicrowaveRAG(
        embeddings=OpenAIEmbeddings(
            model='text-embedding-3-small',
            api_key=SecretStr(OPENAI_API_KEY),
        ),
        llm_client=ChatOpenAI(
            
            temperature=0.0,
            model='gpt-5.2',
            api_key=SecretStr(OPENAI_API_KEY),
        )
    )
)

# - Create OpenAIEmbeddings with model='text-embedding-3-small' and api_key=OPENAI_API_KEY
# - Create ChatOpenAI with temperature=0.0, model='gpt-5.2' and api_key=OPENAI_API_KEY
# - Wrap both in a MicrowaveRAG instance and pass it to main()