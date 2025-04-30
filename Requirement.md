```mermaid
%% XML Embedder App Flow
flowchart TD
    A[User Sends XML via /embed] --> B[Parse XML to Structured JSON]
    B --> C[Generate Embedding using AzureOpenAI]
    C --> D[Store Embedding in FAISS Vector Store File (e.g., index.faiss)]
    D --> E[Store Structured JSON as Metadata in flat_text.pkl]
```

```mermaid
%% XML Search App Flow
flowchart TD
    A[User Sends Query via /search] --> B[LLM Converts Query to Structured JSON]
    B --> C[Generate Embedding from Structured JSON]
    C --> D[Load FAISS Vector Store from index.faiss]
    D --> E[Load Metadata from flat_text.pkl]
    E --> F[Search FAISS Vector Store for Top Matches]
    F --> G[Send Matches + Original Query to LLM]
    G --> H[LLM Filters and Ranks Best Match]
    H --> I[Return Final Matched JSON]
```
