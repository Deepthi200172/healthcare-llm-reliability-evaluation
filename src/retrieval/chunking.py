from typing import Dict, List


def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 64) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    step = max(1, chunk_size - chunk_overlap)
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += step
    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> List[Dict]:
    chunked = []
    for doc in documents:
        for j, chunk in enumerate(chunk_text(doc["content"], chunk_size, chunk_overlap)):
            chunked.append({
                **{k: v for k, v in doc.items() if k != "content"},
                "content": chunk,
                "chunk_id": f"{doc['id']}_c{j}",
            })
    return chunked
