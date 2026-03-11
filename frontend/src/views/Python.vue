<template>
  <div class="python-page">
    <h2>🐍 Python 面试题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="__init__ vs __new__" name="1">
        <div class="answer">
          <p><strong>__new__</strong>: 创建对象时调用，返回实例</p>
          <p><strong>__init__</strong>: 初始化实例时调用，设置属性</p>
          <pre><code>class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="Python GIL 是什么?" name="2">
        <div class="answer">
          <p><strong>GIL (Global Interpreter Lock)</strong>: CPython 的全局解释器锁，保证同一时刻只有一个线程执行 Python 字节码。</p>
          <p><strong>适用场景</strong>:</p>
          <ul>
            <li>✅ I/O 密集型: 多线程</li>
            <li>❌ CPU 密集型: 多进程</li>
          </ul>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="装饰器实现" name="3">
        <div class="answer">
          <pre><code>import functools

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time()-start:.2f}s")
        return result
    return wrapper</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="生成器 vs 迭代器" name="4">
        <div class="answer">
          <p><strong>生成器</strong>: 使用 yield 关键字的函数</p>
          <pre><code>def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# 使用
for num in fibonacci(10):
    print(num)</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="深拷贝 vs 浅拷贝" name="5">
        <div class="answer">
          <pre><code>import copy

# 浅拷贝 - 只拷贝第一层
list2 = copy.copy(list1)

# 深拷贝 - 递归拷贝所有层级
list3 = copy.deepcopy(list1)</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LRU Cache 实现" name="6">
        <div class="answer">
          <pre><code>from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)</code></pre>
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
.python-page {
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

code {
  font-family: 'Consolas', monospace;
}

ul {
  padding-left: 20px;
}
</style>
