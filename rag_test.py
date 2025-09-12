print("imports...")
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

print("load pdf...")
loader = PyPDFLoader("mica.pdf")
pages = loader.load()

print("split...")
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
docs = splitter.split_documents(pages)

# обмежимось, щоб не чекати довго на ембеддінги
docs = docs[:8]
print("docs for index:", len(docs))

print("embeddings...")
emb = OpenAIEmbeddings()  # візьме OPENAI_API_KEY із середовища

print("vector store...")
db = Chroma.from_documents(docs, emb)

print("llm...")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

print("retriever...")
retriever = db.as_retriever()

query = "In one sentence, what is the purpose of the MiCA Regulation?"
print("query:", query)

print("related docs preview:")
for d in retriever.get_relevant_documents(query)[:2]:
    print("-", d.page_content[:160].replace("\n"," "), "...")

print("qa...")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

answer = qa.run(query)
print("\nANSWER:\n", answer)