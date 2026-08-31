from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

def embed_query(query):

    return model.encode(
        query,
        normalize_embeddings=True
    )