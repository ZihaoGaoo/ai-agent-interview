<template>
  <el-container class="app-container">
    <el-header class="header">
      <div class="logo">
        <h1>🤖 AI Agent 面试题库</h1>
        <span class="subtitle">阿里云 AI Agent 研发工程师</span>
      </div>
      <el-menu 
        mode="horizontal" 
        :default-active="$route.path"
        router
        class="nav-menu"
      >
        <el-menu-item index="/">首页</el-menu-item>
        <el-menu-item index="/python">Python</el-menu-item>
        <el-menu-item index="/llm">大模型</el-menu-item>
        <el-menu-item index="/agent">AI Agent</el-menu-item>
        <el-menu-item index="/rag">RAG</el-menu-item>
        <el-menu-item index="/system">系统设计</el-menu-item>
        <el-menu-item index="/coding">代码题</el-menu-item>
      </el-menu>
    </el-header>
    
    <el-main class="main-content">
      <router-view />
    </el-main>
    
    <el-footer class="footer">
      <p>🌾 面试造火箭，工作拧螺丝</p>
    </el-footer>
  </el-container>
</template>

<script setup>
import { ElContainer, ElHeader, ElMain, ElFooter, ElMenu, ElMenuItem } from 'element-plus'
</script>

<style scoped>
.app-container {
  min-height: 100vh;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  padding: 0 20px;
}

.logo h1 {
  color: white;
  margin: 0;
  font-size: 24px;
}

.logo .subtitle {
  color: rgba(255,255,255,0.8);
  font-size: 14px;
}

.nav-menu {
  background: transparent;
  border: none;
}

.main-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

.footer {
  text-align: center;
  color: #666;
  padding: 20px;
}
</style>
