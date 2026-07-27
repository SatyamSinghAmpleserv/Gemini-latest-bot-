
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# # Reading all the existing json
class ReadingFiles:

    # Taking the table names
    def table_file(self):
        import os
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent
        with open(BASE_DIR / "table_schema.json", "r") as f:
            table = json.load(f)
        return table

    

read_files = ReadingFiles()

all_tables = read_files.table_file()


# table_embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
table_embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")


from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def build_table_index(all_tables, embedding_model):

    documents = []

    for table in all_tables['table_schema']:
        print(table['table_name'])
        doc = Document(
            page_content=(
                f"Table: {table['table_name']}\n"
                f"Description: {table['description']}"
            ),
            metadata={
                "type": table['type'],
                "table": table['table_name'],
                "description": table['description']
            }
        )

        documents.append(doc)

    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )

    vector_store.save_local("table_faiss_index")
    print('Embedding saved!')
    return vector_store



table_retrieval = build_table_index(all_tables=all_tables, embedding_model=table_embedding_model)
# query = input('Ask anything:')
# results = VECTOR_STORE.similarity_search_with_score(query=query, k=5)
# print("="*100)
# print(results)
# print("="*100)

# def build_schema_index():
#     return 