from happy_hub.storage import SimpleVectorDB


if __name__ == "__main__":
    new_db = SimpleVectorDB()
    new_db.load_from_file("vector_db.pkl")
    
    results = new_db.search("tín dụng xanh ở MB là gì ?", top_k=3)
    document = []
    for doc, score in results:
        print(doc)
        print('-'*300)