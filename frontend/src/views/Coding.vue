<template>
  <div class="coding-page">
    <h2>💻 手写代码题</h2>
    
    <el-collapse v-model="activeNames">
      <el-collapse-item title="两数之和" name="1">
        <div class="answer">
          <pre><code>def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="反转链表" name="2">
        <div class="answer">
          <pre><code>def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="快速排序" name="3">
        <div class="answer">
          <pre><code>def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="合并两个有序数组" name="4">
        <div class="answer">
          <pre><code>def merge(nums1, m, nums2, n):
    p1, p2 = m - 1, n - 1
    p = m + n - 1
    
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1
    
    nums1[:p2 + 1] = nums2[:p2 + 1]</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="二叉树遍历" name="5">
        <div class="answer">
          <pre><code># 前序遍历
def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

# 中序遍历  
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

# 后序遍历
def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]</code></pre>
        </div>
      </el-collapse-item>
      
      <el-collapse-item title="LRU 缓存" name="6">
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
.coding-page {
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
  font-size: 14px;
}
</style>
