import os
import pickle
import faiss
import param

# import faiss
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# from langchain.document_loaders import UnstructuredURLLoader

os.environ["OPENAI_API_KEY"] = "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"


# urls = [
#     "https://spendid.io/",
#     "https://spendid.io/terms-of-service",
#     "https://spendid.io/privacy-policy",
#     "https://spendid.io/web-app-summary",
#     "https://spendid.io/faqs",
#     "https://spendid.io/about",
#     "https://spendid.io/contact",
#     "https://spendid.io/faqs",
#     "https://spendid.io/api-demo",
#     "https://mybudgetreport.com/financial-advisors",
#     "https://mybudgetreport.com/employers",
# ]
# loaders = UnstructuredURLLoader(urls=urls)
# data = loaders.load()

# # Text Splitter

# text_splitter = CharacterTextSplitter(
#     separator="\n", chunk_size=1000, chunk_overlap=200
# )


# docs = text_splitter.split_documents(data)


# embeddings = OpenAIEmbeddings()


def load_db():
    with open("faiss_store_openai.pkl", "rb") as f:
        store = pickle.load(f)
    retriever = store.as_retriever(search_kwargs=dict(k=3))

    # llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa


class Chatbot(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query = param.String("")
    db_response = param.List([])

    def __init__(self, **params):
        super(Chatbot, self).__init__(**params)
        self.qa = load_db()

    def convchain(self, query):
        conversation = []

        if not query:
            user_response = {"User": "(No query provided)"}
            chatbot_response = {"ChatBot": "(No response)"}
        else:
            result = self.qa({"question": query, "chat_history": self.chat_history})
            self.chat_history.extend([(query, result["answer"])])
            self.db_query = result["generated_question"]
            self.db_response = result["source_documents"]
            self.answer = result["answer"]
            user_response = {"User": query}
            chatbot_response = {"ChatBot": self.answer}

        conversation.append(user_response)
        conversation.append(chatbot_response)
        return self.answer

    def clr_history(self):
        self.chat_history = []
        return


cb = Chatbot()


while True:
    user_input = input("User: ")
    response = cb.convchain(user_input)
    print("Chatbot: ", response)

    if user_input.lower() in ["exit", "quit"]:
        break
