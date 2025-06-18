# app.py
import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
import traceback
from dotenv import load_dotenv

# Load env
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    logger.error("API_KEY not set; endpoint will fail until set as Vercel env var.")

# FastAPI app
app = FastAPI(title="RAG Query API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # base64 image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# DB connection per request
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"DB connect error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

# Cosine similarity
def cosine_similarity(vec1, vec2):
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        if v1.size == 0 or v2.size == 0:
            return 0.0
        dp = np.dot(v1, v2)
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(dp / (n1 * n2))
    except Exception as e:
        logger.error(f"cosine_similarity error: {e}")
        return 0.0

# Get embedding via aipipe proxy
async def get_embedding(text, max_retries=3):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    retries = 0
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    async with aiohttp.ClientSession() as session:
        while retries < max_retries:
            try:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["data"][0]["embedding"]
                    elif resp.status == 429:
                        await asyncio.sleep(5 * (retries + 1))
                        retries += 1
                    else:
                        text_err = await resp.text()
                        logger.error(f"Embedding API error {resp.status}: {text_err}")
                        raise HTTPException(status_code=resp.status, detail="Embedding API error")
            except Exception as e:
                logger.error(f"Exception in get_embedding: {e}")
                retries += 1
                await asyncio.sleep(3 * retries)
        raise HTTPException(status_code=500, detail="Failed to get embedding")

# Multimodal: handle optional image
async def process_multimodal_query(question, image_base64):
    if image_base64:
        # Try vision endpoint
        try:
            # Example: send image as data URL
            image_content = f"data:image/jpeg;base64,{image_base64}"
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Describe relevant aspects of this image for question: {question}"},
                            {"type": "image_url", "image_url": {"url": image_content}}
                        ]
                    }
                ]
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        res = await resp.json()
                        desc = res["choices"][0]["message"]["content"]
                        combined = question + "\nImage context: " + desc
                        return await get_embedding(combined)
                    else:
                        logger.warning("Image processing failed, fallback to text-only")
                        return await get_embedding(question)
        except Exception as e:
            logger.error(f"Multimodal error: {e}")
            return await get_embedding(question)
    else:
        return await get_embedding(question)

# Find similar chunks in SQLite
async def find_similar_content(query_embedding, conn):
    try:
        c = conn.cursor()
        results = []
        # Discourse chunks
        c.execute("SELECT id, post_id, topic_id, topic_title, post_number, author, created_at, likes, chunk_index, content, url, embedding FROM discourse_chunks WHERE embedding IS NOT NULL")
        rows = c.fetchall()
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_embedding, emb)
                if sim >= SIMILARITY_THRESHOLD:
                    url = row["url"]
                    if not url.startswith("http"):
                        url = f"https://discourse.onlinedegree.iitm.ac.in{url}"
                    results.append({
                        "source": "discourse",
                        "id": row["id"],
                        "post_id": row["post_id"],
                        "topic_id": row["topic_id"],
                        "title": row["topic_title"],
                        "url": url,
                        "content": row["content"],
                        "chunk_index": int(row["chunk_index"]) if row["chunk_index"].isdigit() else row["chunk_index"],
                        "similarity": sim
                    })
            except Exception as e:
                logger.error(f"Error processing discourse row id={row['id']}: {e}")
        # Markdown chunks
        c.execute("SELECT id, doc_title, original_url, downloaded_at, chunk_index, content, embedding FROM markdown_chunks WHERE embedding IS NOT NULL")
        rows = c.fetchall()
        for row in rows:
            try:
                emb = json.loads(row["embedding"])
                sim = cosine_similarity(query_embedding, emb)
                if sim >= SIMILARITY_THRESHOLD:
                    url = row["original_url"]
                    if not url or not url.startswith("http"):
                        url = f"https://docs.onlinedegree.iitm.ac.in/{row['doc_title']}"
                    results.append({
                        "source": "markdown",
                        "id": row["id"],
                        "title": row["doc_title"],
                        "url": url,
                        "content": row["content"],
                        "chunk_index": int(row["chunk_index"]) if row["chunk_index"].isdigit() else row["chunk_index"],
                        "similarity": sim
                    })
            except Exception as e:
                logger.error(f"Error processing markdown row id={row['id']}: {e}")
        # Sort & group
        results.sort(key=lambda x: x["similarity"], reverse=True)
        grouped = {}
        for r in results:
            key = f"{r['source']}_{r.get('post_id', r.get('title'))}"
            grouped.setdefault(key, []).append(r)
        final = []
        for chunks in grouped.values():
            chunks.sort(key=lambda x: x["similarity"], reverse=True)
            final.extend(chunks[:MAX_CONTEXT_CHUNKS])
        final.sort(key=lambda x: x["similarity"], reverse=True)
        return final[:MAX_RESULTS]
    except Exception as e:
        logger.error(f"find_similar_content error: {e}")
        raise

# Enrich with adjacent chunks
async def enrich_with_adjacent_chunks(conn, results):
    try:
        c = conn.cursor()
        enriched = []
        for res in results:
            cont = res["content"]
            add = ""
            if res["source"] == "discourse":
                pid = res["post_id"]
                idx = res["chunk_index"]
                if isinstance(idx, int) and idx > 0:
                    c.execute("SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?", (pid, str(idx-1)))
                    prev = c.fetchone()
                    if prev:
                        add += prev["content"] + " "
                c.execute("SELECT content FROM discourse_chunks WHERE post_id = ? AND chunk_index = ?", (pid, str(idx+1)))
                nxt = c.fetchone()
                if nxt:
                    add += nxt["content"]
            else:
                title = res["title"]
                idx = res["chunk_index"]
                if isinstance(idx, int) and idx > 0:
                    c.execute("SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?", (title, str(idx-1)))
                    prev = c.fetchone()
                    if prev:
                        add += prev["content"] + " "
                c.execute("SELECT content FROM markdown_chunks WHERE doc_title = ? AND chunk_index = ?", (title, str(idx+1)))
                nxt = c.fetchone()
                if nxt:
                    add += nxt["content"]
            if add:
                res["content"] = cont + " " + add
            enriched.append(res)
        return enriched
    except Exception as e:
        logger.error(f"enrich error: {e}")
        raise

# Generate answer via LLM
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    retries = 0
    while retries < max_retries:
        try:
            # Build context block
            context = ""
            for r in relevant_results:
                stype = "Discourse post" if r["source"]=="discourse" else "Documentation"
                snippet = r["content"][:1500].strip()
                context += f"\n\n{stype} (URL: {r['url']}):\n{snippet}"
            prompt = (
                "Answer the following question based ONLY on the provided context.\n"
                "If you cannot answer, say \"I don't have enough information to answer this question.\"\n\n"
                "Context:\n"
                f"{context}\n\n"
                f"Question: {question}\n\n"
                "Return in exact format:\n"
                "1. A comprehensive yet concise answer\n"
                "2. A \"Sources:\" section listing URLs and brief text excerpts used\n\n"
                "Sources must be:\n"
                "Sources:\n"
                "1. URL: [exact_url], Text: [brief quote]\n"
                "2. URL: [exact_url], Text: [brief quote]\n"
                "Make sure URLs are copied exactly from context."
            )
            url = "https://aipipe.org/openai/v1/chat/completions"
            headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant answering based only on provided context. Include exact sources."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3
            }
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["choices"][0]["message"]["content"]
                    elif resp.status == 429:
                        await asyncio.sleep(3 * (retries+1))
                        retries += 1
                    else:
                        txt = await resp.text()
                        logger.error(f"LLM API error {resp.status}: {txt}")
                        raise HTTPException(status_code=resp.status, detail="LLM API error")
        except Exception as e:
            logger.error(f"generate_answer exception: {e}")
            retries += 1
            await asyncio.sleep(2)
    raise HTTPException(status_code=500, detail="Failed to generate answer")

# Parse LLM response
def parse_llm_response(response: str):
    try:
        parts = response.split("Sources:", 1)
        if len(parts) == 1:
            for hd in ["Source:", "References:", "Reference:"]:
                if hd in response:
                    parts = response.split(hd, 1)
                    break
        answer = parts[0].strip()
        links = []
        if len(parts) > 1:
            src_text = parts[1].strip()
            lines = src_text.splitlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                line = re.sub(r'^\d+\.\s*', '', line)
                line = re.sub(r'^-\s*', '', line)
                # URL extraction
                url_m = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|URL:\s*(http\S+)|\[(http\S+)\]|(http\S+)', line, re.IGNORECASE)
                text_m = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
                if url_m:
                    groups = url_m.groups()
                    url = next((g for g in groups if g), "").strip()
                    text = "Source reference"
                    if text_m:
                        t_groups = text_m.groups()
                        txt = next((g for g in t_groups if g), "")
                        if txt:
                            text = txt.strip()
                    if url and url.startswith("http"):
                        links.append({"url": url, "text": text})
        return {"answer": answer, "links": links}
    except Exception as e:
        logger.error(f"parse_llm_response error: {e}")
        return {"answer": "Error parsing LLM response.", "links": []}

# API endpoint
@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    logger.info(f"Received query: '{request.question[:50]}...' image? {request.image is not None}")
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    conn = get_db_connection()
    try:
        # Embedding
        query_emb = await process_multimodal_query(request.question, request.image)
        # Retrieval
        relevant = await find_similar_content(query_emb, conn)
        if not relevant:
            return {"answer": "I couldn't find relevant info in my knowledge base.", "links": []}
        # Enrich
        enriched = await enrich_with_adjacent_chunks(conn, relevant)
        # Generate answer
        llm_resp = await generate_answer(request.question, enriched)
        parsed = parse_llm_response(llm_resp)
        if not parsed["links"]:
            # fallback links from top chunks
            links = []
            seen = set()
            for r in relevant[:5]:
                url = r["url"]
                if url not in seen:
                    seen.add(url)
                    snippet = r["content"][:100] + ("..." if len(r["content"])>100 else "")
                    links.append({"url": url, "text": snippet})
            parsed["links"] = links
        logger.info(f"Returning answer length={len(parsed['answer'])}, links={len(parsed['links'])}")
        return parsed
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /query: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Internal error processing query")
    finally:
        conn.close()

# Health
@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM discourse_chunks")
        dc = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM markdown_chunks")
        mc = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        dec = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        mec = c.fetchone()[0]
        conn.close()
        return {
            "status": "healthy",
            "db_connected": True,
            "api_key_set": bool(API_KEY),
            "discourse_chunks": dc,
            "markdown_chunks": mc,
            "discourse_embeddings": dec,
            "markdown_embeddings": mec
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)}

# For local dev
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
