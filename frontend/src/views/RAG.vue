<template>
  <div class="rag-page">
    <h2>📚 RAG (检索增强生成) 面试题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="什么是 RAG?" name="1">
        <div class="answer">
          <p><strong>RAG = Retrieval + Generation</strong></p>
          <p>解决的问题:</p>
          <ul>
            <li>知识时效性 - LLM 训练数据截止日期</li>
            <li>领域知识 - 专业领域知识不足</li>
            <li>Hallucination - 减少模型幻觉</li>
          </ul>
          <pre><code>User Query → Retriever → Context → LLM → Answer</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="向量检索原理" name="2">
        <div class="answer">
          <p>将文本转为向量,在向量空间中找相似文本</p>
          <pre><code>from sklearn.metrics.pairwise import cosine_similarity

def vector_search(query_embedding, doc_embeddings, top_k=5):
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [(idx, similarities[idx]) for idx in top_indices]</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="Hybrid Search" name="3">
        <div class="answer">
          <p><strong>混合检索</strong> = 关键词检索 + 向量检索</p>
          <pre><code>from langchain.retrievers import EnsembleRetriever

ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="Reranking 两阶段检索" name="4">
        <div class="answer">
          <p>粗排 → 精排</p>
          <pre><code># 第一阶段: 向量检索 (Top 100)
candidates = vector_retriever.get_relevant_documents(query)[:100]

# 第二阶段: Cross-Encoder 重排序
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
scores = reranker.predict([(query, doc) for doc in candidates])
ranked = sorted(zip(candidates, scores), key=lambda x: x[1])[:5]</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="分块策略" name="5">
        <div class="answer">
          <table>
            <tr><th>方法</th><th>描述</th></tr>
            <tr><td>固定大小</td><td>按字符/词数量切分</td></tr>
            <tr><td>句子级</td><td>按句子切分</td></tr>
            <tr><td>递归分块</td><td>按层级递归切分</td></tr>
            <tr><td>语义分块</td><td>按语义边界切分</td></tr>
          </table>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="RAG 优化方法" name="6">
        <div class="answer">
          <ul>
            <li><strong>Query 改写</strong>: HyDE、Query Expansion</li>
            <li><strong>Chunk 策略</strong>: Small-to-large</li>
            <li><strong>排序</strong>: Cross-Encoder 重排</li>
            <li><strong>混合搜索</strong>: BM25 + 向量</li>
          </ul>
        </div>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElCollapse, ElCollapseItem } from 'element-plus'

const activeNames = ref(['1'])
</script>

<style scoped>
.rag-page {
  padding: 20px;
}

.answer {
  padding: 10px;
}

pre {
  background: #f5f5f5;
  padding: 15px;
  border-radius: 5px;
  overflow-x: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
}

th, td {
  border: 1px solid #ddd;
  padding: 8px;
  text-align: center;
}

th {
  background: #f5f5f5;
}

ul {
  padding-left: 20px;
}
</style>
