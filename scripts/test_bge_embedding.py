from app.vectorstore.embeddings import BGEFlagEmbeddingModel


def main() -> None:
    model = BGEFlagEmbeddingModel()

    sentence = "Şirketin net karı 2024 yılında yüzde 18.4 arttı."
    embedding = model.embed_text(sentence)

    print("Single text embedding")
    print("Type:", type(embedding))
    print("Dimension:", len(embedding))
    print("First 5 values:", embedding[:5])
    print()

    sentences = [
        "Faiz giderleri bu çeyrekte azaldı.",
        "Nakit akışı geçen yıla göre güçlendi.",
    ]
    embeddings = model.embed_documents(sentences)

    print("Batch embedding")
    print("Embedding count:", len(embeddings))
    print("First embedding dimension:", len(embeddings[0]))
    print("First 5 values of second embedding:", embeddings[1][:5])


if __name__ == "__main__":
    main()