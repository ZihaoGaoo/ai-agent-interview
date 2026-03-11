<template>
  <div class="system-page">
    <h2">🏗️ 系统设计面试题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="分布式系统的 CAP 定理" name="1">
        <div class="answer">
          <p><strong>CAP</strong>: Consistency(一致性) + Availability(可用性) + Partition tolerance(分区容错)</p>
          <p>分布式系统只能满足其中两个:</p>
          <ul>
            <li><strong>CA</strong>: 单点数据库</li>
            <li><strong>CP</strong>: Redis, HBase</li>
            <li><strong>AP</strong>: Cassandra, DynamoDB</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="BASE 理论" name="2">
        <div class="answer">
          <ul>
            <li><strong>Basically Available</strong>: 基本可用</li>
            <li><strong>Soft state</strong>: 软状态</li>
            <li><strong>Eventually consistent</strong>: 最终一致</li>
          </ul>
          <p>是对 CAP 中 AP 的补充</p>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="负载均衡算法" name="3">
        <div class="answer">
          <ul>
            <li><strong>Round Robin</strong>: 轮询</li>
            <li><strong>Weighted Round Robin</strong>: 加权轮询</li>
            <li><strong>Least Connections</strong>: 最少连接</li>
            <li><strong>IP Hash</strong>: 源 IP 哈希</li>
            <li><strong>Random</strong>: 随机</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="数据库索引" name="4">
        <div class="answer">
          <ul>
            <li><strong>B+ Tree</strong>: MySQL InnoDB</li>
            <li><strong>Hash Index</strong>: Memory DB</li>
            <li><strong>Full Text</strong>: 全文索引</li>
          </ul>
          <p><strong>最左前缀原则</strong>: 联合索引遵循最左匹配</p>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="缓存策略" name="5">
        <div class="answer">
          <ul>
            <li><strong>Cache-Aside</strong>: 应用自己管理</li>
            <li><strong>Read-Through</strong>: 缓存加载数据</li>
            <li><strong>Write-Through</strong>: 同步写</li>
            <li><strong>Write-Behind</strong>: 异步写</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="消息队列作用" name="6">
        <div class="answer">
          <ul>
            <li><strong>解耦</strong>: 生产者消费者分离</li>
            <li><strong>削峰</strong>: 流量削峰填谷</li>
            <li><strong>异步</strong>: 异步处理</li>
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
.system-page {
  padding: 20px;
}

.answer {
  padding: 10px;
}

ul {
  padding-left: 20px;
}
</style>
