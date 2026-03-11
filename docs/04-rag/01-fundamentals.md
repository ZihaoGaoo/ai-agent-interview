# RAG (检索增强生成) 面试题汇总

## 1. RAG 基础

### Q1: 什么是 RAG? 为什么需要 RAG?

**RAG = Retrieval + Generation**

解决的问题:
- **知识时效性**: LLM 训练数据截止日期
- **领域知识**: 专业领域知识不足
- ** hallucination**: 减少模型幻觉
- **可控性**: 可更新、可追溯

```
┌─────────────────────────────────────────────┐
│                  RAG 流程                    │
├─────────────────────────────────────────────┤
│                                             │
│  User Query                                 │
│       │                                     │
│       ▼                                     │
│  ┌─────────────┐                            │
│  │  Retriever  │ ← 向量检索 (Embedding)     │
│  └─────────────┘                            │
│       │                                     │
│       ▼                                     │
│  ┌─────────────┐                            │
│  │   Context   │ ← 相关文档                 │
│  └─────────────┘                            │
│       │                                     │
│       ▼                                     │
│  ┌─────────────┐                            │
│  │    LLM      │ ← 生成答案                 │
│  └─────────────┘                            │
│       │                                     │
│       ▼                                     │
│     Answer                                  │
│                                             │
└─────────────────────────────────────────────┘
```

### Q2: RAG 的核心组件?

| 组件 | 技术 |
|------|------|
| **分词/向量化** | BGE, BAAI, OpenAI Embedding |
| **向量数据库** | Milvus, Pinecone, Qdrant, Chroma |
| **检索器** | BM25, Dense Retriever, Hybrid |
| **排序器** | Cross-Encoder, Learning to Rank |

```python
# 基础 RAG 实现
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. 创建向量存储
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# 2. 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# 3. 执行
result = qa_chain.run("什么是 RAG?")
```

## 2. 检索技术

### Q3: 向量检索的原理?

将文本转为向量,在向量空间中找相似文本。

```python
# 简单的向量检索
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def vector_search(query_embedding, doc_embeddings, top_k=5):
    # 计算余弦相似度
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    
    # 取 top_k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    return [(idx, similarities[idx]) for idx in top_indices]
```

### Q4: 什么是 Hybrid Search?

混合检索 = 关键词检索 + 向量检索

```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.vectorstores import FAISS

# 1. BM25 检索
bm25_retriever = BM25Retriever.from_documents(documents)

# 2. 向量检索  
faiss_store = FAISS.from_documents(documents, embeddings)
vector_retriever = faiss_store.as_retriever()

# 3. 混合检索
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

### Q5: 什么是 Reranking?

两阶段检索:粗排 → 精排

```python
from sentence_transformers import CrossEncoder

# 第一阶段: 向量检索 (Top 100)
candidates = vector_retriever.get_relevant_documents(query)[:100]

# 第二阶段: Cross-Encoder 重排序
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc.page_content) for doc in candidates])

# 排序取 Top 5
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)[:5]
```

## 3. 分块策略

### Q6: 如何对文档进行分块?

| 方法 | 描述 | 适用场景 |
|------|------|----------|
| **固定大小** | 按字符/词数量切分 | 通用 |
| **句子级** | 按句子切分 | 保留语义 |
| **递归分块** | 按层级递归切分 | 结构化文档 |
| **语义分块** | 按语义边界切分 | 高质量分割 |

```python
# 固定大小分块
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 句子级分块
from langchain.text_splitter import SentenceTextSplitter
splitter = SentenceTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(text)
```

### Q7: 如何选择 chunk size?

- **小 chunk** (256-512): 精确但可能缺少上下文
- **中 chunk** (512-1024): 平衡
- **大 chunk** (1024+): 上下文丰富但可能引入噪声

**最佳实践**: 使用 **small-to-large** 策略检索

```python
# Small-to-large: 先检索小chunk,再扩展到父chunk
class ParentDocumentRetriever:
    def __init__(self, child_splitter, parent_splitter):
        self.child_splitter = child_splitter
        self.parent_splitter = parent_splitter
    
    def get_relevant_documents(self, query):
        # 1. 检索小chunk
        child_docs = self.child_retriever.get_relevant_documents(query)
        
        # 2. 获取对应的父文档
        parent_docs = []
        for child in child_docs:
            parent = self.find_parent(child)
            parent_docs.append(parent)
        
        return parent_docs
```

## 4. RAG 优化

### Q8: RAG 常见的优化方法?

| 优化点 | 方法 |
|--------|------|
| **检索质量** | Query 改写、HyDE、Expansion |
| **Chunk 策略** | Small-to-large、滑动窗口 |
| **排序** | Cross-Encoder 重排 |
| **上下文** | 上下文压缩、过滤 |
| **混合搜索** | BM25 + 向量 |

```python
# Query 改写
class QueryRewriter:
    def __init__(self, llm):
        self.llm = llm
    
    def rewrite(self, query):
        prompt = f"请将以下查询改写为更适合检索的版本:\n{query}"
        return self.llm.generate(prompt)

# HyDE (Hypothetical Document Embeddings)
hyde_prompt = """
请生成一个可能包含答案的假设文档:
{query}

假设文档:
"""
```

### Q9: 什么是 RagFusion?

多查询检索 + 融合排序

```python
# 生成多个查询
queries = [
    original_query,
    llm.generate(f"简化: {query}"),
    llm.generate(f"专业术语: {query}"),
    llm.generate(f"例子: {query}")
]

# 检索并融合
all_docs = []
for q in queries:
    docs = retriever.get_relevant_documents(q)
    all_docs.extend(docs)

# MMR 重排序
final_docs = mmr_rerank(all_docs, query)
```

## 5. 向量数据库

### Q10: 主流向量数据库对比?

| 数据库 | 特点 | 适用场景 |
|--------|------|----------|
| **Pinecone** | 托管服务、易用 | 快速上线 |
| **Milvus** | 开源、功能全 | 大规模 |
| **Qdrant** | 高性能、Rust | 低延迟 |
| **Chroma** | 轻量、易用 | 本地/小规模 |
| **Weaviate** | 图结构、GraphQL | 复杂查询 |

```python
# Milvus 示例
from pymilvus import connections, Collection

# 连接
connections.connect(host='localhost', port='19530')
collection = Collection('my_collection')

# 检索
search_params = {"metric_type": "IP", "params": {}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=10
)
```

---

> 参考: LangChain RAG, LlamaIndex, RAG 论文
