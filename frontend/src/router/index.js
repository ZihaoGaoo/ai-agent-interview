import { createRouter, createWebHistory } from 'vue-router'
import Home from '../views/Home.vue'
import Python from '../views/Python.vue'
import LLM from '../views/LLM.vue'
import Agent from '../views/Agent.vue'
import RAG from '../views/RAG.vue'
import System from '../views/System.vue'
import Coding from '../views/Coding.vue'

const routes = [
  { path: '/', name: 'Home', component: Home },
  { path: '/python', name: 'Python', component: Python },
  { path: '/llm', name: 'LLM', component: LLM },
  { path: '/agent', name: 'Agent', component: Agent },
  { path: '/rag', name: 'RAG', component: RAG },
  { path: '/system', name: 'System', component: System },
  { path: '/coding', name: 'Coding', component: Coding },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router
