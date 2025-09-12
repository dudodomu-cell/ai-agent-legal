print("start")

from langchain_community.document_loaders import PyPDFLoader

print("loader ok")

loader = PyPDFLoader("mica.pdf")
pages = loader.load()

print("pages:", len(pages))
print("sample:", pages[0].page_content[:200].replace("\n"," ") if pages else "NO PAGES")