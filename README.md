# Mental Health Assistant (RAG-Based Chatbot)

An AI-powered conversational system designed to provide supportive responses for mental health-related queries using Retrieval-Augmented Generation (RAG).

---

## Overview

This project combines semantic search with intelligent response generation to build a context-aware mental health assistant.

Instead of generating generic answers, the system:
- Retrieves relevant information from a dataset
- Understands user intent
- Generates structured and meaningful responses

---

## Key Features

### Retrieval-Augmented Generation (RAG)
- Retrieves relevant context before generating responses
- Improves accuracy compared to standard chatbots

### Semantic Search
- Uses sentence-transformers for embedding-based similarity
- Enables meaning-based retrieval instead of keyword matching

### Intent Detection
- Classifies queries into:
  - Causes (why / reason)
  - Advice (what to do / how to handle)
  - Symptoms (feeling / experiencing)

### Conversational Interface
- Built using Streamlit
- Provides a chat-based user experience

### RAG Debugger
- Displays:
  - Confidence score
  - Retrieved documents
  - Retrieval quality analysis
- Suggests improvements such as:
  - Increasing dataset coverage
  - Improving embedding methods

---

## System Architecture

User Query → Embedding → Similarity Search → Context Retrieval → Response Generation

---

## Tech Stack

- Frontend: Streamlit  
- Backend: Python  
- NLP Model: sentence-transformers  
- Similarity Metric: Cosine similarity  
- Data Source: Custom text datasets  

---

## How It Works

1. The user enters a query (e.g., "How to handle overthinking?")
2. The query is converted into vector embeddings
3. The system retrieves top-K relevant documents
4. The query intent is detected
5. A response is generated using:
   - Retrieved context
   - Intent classification
6. The system provides debugging insights on retrieval performance

---

## Example Queries

- Why do I feel anxious suddenly?
- How to handle overthinking?
- What causes panic attacks?
- Signs of depression

---

## Limitations

- Works on a limited dataset (not full-scale knowledge)
- Does not replace professional mental health support
- Uses a lightweight architecture without external LLM APIs

---

## Future Improvements

- Integration with large language models (OpenAI, Gemini, etc.)
- Use of vector databases (FAISS, Pinecone)
- Expansion of dataset size and diversity
- Personalized responses based on user history

---

## Demo

Add screenshots or a short demo video here.

---

## Disclaimer

This system provides general guidance only and is not intended as a medical diagnosis tool.

---

## Conclusion

This project demonstrates:
- Understanding of Retrieval-Augmented Generation (RAG)
- Practical use of semantic embeddings
- Ability to build end-to-end AI applications
- Focus on explainability and user experience
