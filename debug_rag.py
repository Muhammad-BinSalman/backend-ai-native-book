import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("QDRANT_API_ENDPOINT")
api_key = os.getenv("QDRANT_API_KEY")
collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_chunks")

print(f"Connecting to {endpoint}...")
client = QdrantClient(url=endpoint, api_key=api_key)

try:
    info = client.get_collection(collection_name)
    print(f"Collection info: {info}")
    
    # Try accessing points_count directly if available, or just print the whole object
    if hasattr(info, 'points_count'):
         print(f"Points count: {info.points_count}")
         
         if info.points_count == 0:
            print("\nWARNING: Collection is empty! You need to ingest the book content.")
         else:
            print("\nCollection has data.")
            scroll_result = client.scroll(
                collection_name=collection_name,
                limit=1,
                with_payload=True,
                with_vectors=False
            )
            val = scroll_result[0]
            if val:
                print(f"Sample point: {val[0].payload}")
            else:
                print("Could not scroll any points.")
    else:
        print("Could not find points_count attribute.")

    
    # Test Search
    print("\nTesting Search...")
    from app.services.cohere_service import cohere_service
    import asyncio
    
    async def test_search():
        query = "What is AI-Native?"
        print(f"Query: {query}")
        embedding = await cohere_service.embed_text(query)
        print(f"Embedding length: {len(embedding)}")
        print(f"Client attributes: {dir(client)}")
        
        results = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=3
        )
        print(f"Search found {len(results)} results")
        for res in results:
            print(f"- Score: {res.score:.4f}, Text: {res.payload.get('text')[:100]}...")

    asyncio.run(test_search())

except Exception as e:
    print(f"Error: {e}")
