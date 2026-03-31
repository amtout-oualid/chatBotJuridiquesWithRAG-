from langchain_pinecone import PineconeVectorStore
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embedding,
    index_name=index_name
)