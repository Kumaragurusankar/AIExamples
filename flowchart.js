A[Start] --> B[embedding_generator.py]
    B --> B1[Load XMLs from csv]
    B1 --> B2[Parse XMLs into flat dictionaries]
    B2 --> B3[Create embeddings]
    B3 --> B4[Build FAISS index]
    B4 --> B5[Save faiss.index and flat_texts.pkl]
    
    B5 --> C{New XMLs?}
    
    C -- Yes --> D[embedding_updater.py (Optional Future)]
    D --> D1[Monitor folder for new XMLs]
    D1 --> D2[Parse and embed new XMLs]
    D2 --> D3[Add new vectors to FAISS index]
    D3 --> D4[Update flat_texts.pkl]
    D4 --> E[app.py (Streamlit)]
    
    C -- No --> E[app.py (Streamlit)]

    E --> E1[Load faiss.index and flat_texts.pkl]
    E1 --> E2[Accept user query input]
    E2 --> E3[Generate structured query (LLM Chain)]
    E3 --> E4[Embed the query]
    E4 --> E5[Search FAISS for top matches]
    E5 --> E6[Display results in app UI]
