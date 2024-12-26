<template>
  <div class='container'>
    <div class='left-column'>
      <div class='header'>
        <div class='text-container'>
          <h1>会议纪要助手</h1>
          <p>一款 AI 驱动的工具</p>
          <p>用于快速将会议录音转换为会议纪要</p>
        </div>
        <el-icon class='icon' size='50' color='#9FD6DC' style='padding-top: 20px'>
          <DataAnalysis/>
        </el-icon>
      </div>
      <ul>
        <li v-for='(file, index) in files' :key='index'>
          <div>
            <div class='file-name'>{{ file.name }}</div>
            <div class='file-date'>{{ file.date }}</div>
          </div>
          <div>
            <el-icon size='20' @click='downloadFile' style='margin-right: 8px'>
              <Download class='download-icon'/>
            </el-icon>
            <el-icon size='20' @click='deleteFile(index)'>
              <Delete class='delete-icon'/>
            </el-icon>
          </div>
        </li>
      </ul>
      <div class='button-container'>
        <label class='upload-button' @click='clearAllFiles'>
          <el-icon class='plus-delete-icon'>
            <Delete/>
          </el-icon>
          清空全部
        </label>
        <input type='file' id='fileInput' style='display: none' @change='handleFileUpload'/>
        <label for='fileInput' class='upload-button'>
          <el-icon class='plus-delete-icon'>
            <Plus/>
          </el-icon>
          上传会议录音
        </label>
      </div>
    </div>
    <div class='right-column'>
      <div v-if='isLoading' class='spinner-container'>
        <div class='spinner'/>
        <p>正在生成会议纪要...</p>
      </div>
      <div v-else-if='files.length > 0' v-html='renderedMarkdown' class='markdown-content'/>
      <div v-else class='initial-prompt'>
        <p>请上传会议录音以生成会议纪要</p>
      </div>
    </div>
  </div>
</template>

<script setup lang='ts'>
import {ref} from 'vue'
import {DataAnalysis, Delete, Download, Plus} from '@element-plus/icons-vue'
import {marked} from 'marked'

interface FileItem {
  name: string
  date: string
}

const files = ref<FileItem[]>([])
const renderedMarkdown = ref('')
const isLoading = ref(false)
let loadingTimeout: ReturnType<typeof setTimeout> | null = null

function handleFileUpload(event: Event) {
  const target = event.target as HTMLInputElement
  const file = target.files?.[0]
  if (file) {
    const currentDate = new Date().toLocaleString()
    files.value.push({name: file.name, date: currentDate})
    triggerLoading()
  }
}

function triggerLoading() {
  if (isLoading.value) {
    if (loadingTimeout) {
      clearTimeout(loadingTimeout)
    }
  } else {
    isLoading.value = true
  }
  loadingTimeout = setTimeout(() => {
    fetchAndRenderMarkdown()
    isLoading.value = false
    loadingTimeout = null
  }, 5000)
}

function clearAllFiles() {
  files.value = []
}

function deleteFile(index: number) {
  files.value.splice(index, 1)
}

async function fetchAndRenderMarkdown() {
  try {
    const response = await fetch('/meeting-minutes.md')
    const markdown = await response.text()
    // noinspection TypeScriptValidateTypes
    renderedMarkdown.value = marked(markdown)
  } catch (error) {
    renderedMarkdown.value = '<p>无法生成会议纪要</p>'
  }
}

async function downloadFile() {
  try {
    const response = await fetch('/meeting-minutes.pdf', {
      method: 'GET',
      headers: {
        'Content-Type': 'text/markdown',
      },
    })
    const blob = await response.blob()
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'meeting-minutes.pdf'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  } catch (error) {
  }
}
</script>

<style scoped lang='css'>
* {
  user-select: none;
  -webkit-user-drag: none;
}

.container {
  margin: 30px 22px;
  display: flex;
  height: calc(100vh - 64px);
  width: calc(100vw - 64px);
  border: 2px solid #86C8D6;
  border-radius: 16px;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.left-column {
  flex: 0 0 400px;
  padding: 20px;
  width: 400px;
  background-color: #E7F8FF;
  border-right: 2px solid #9FD6DC;
  display: flex;
  flex-direction: column;
}

.right-column {
  flex-grow: 1;
  padding: 20px;
  overflow-y: auto;
  background-color: #FFFFFF;
}

h1 {
  color: #333333;
}

p {
  margin: 0;
  line-height: 1.8;
}

ul {
  list-style-type: none;
  padding: 0;
  flex-grow: 1;
  overflow-y: auto;
}

li {
  padding: 14px;
  margin-bottom: 12px;
  background-color: white;
  border: 3px solid #5AA1AF;
  border-radius: 16px;
  color: #333333;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.file-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.file-date {
  font-size: 14px;
  color: #666666;
}

.delete-icon {
  cursor: pointer;
  color: #FF4D4F;
}

.download-icon {
  cursor: pointer;
  color: #409EFF;
}

.button-container {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 20px;
  gap: 40px;
}

.upload-button {
  padding: 10px 20px;
  background-color: white;
  border: 2px solid #5AA1AF;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  border-radius: 16px;
  cursor: pointer;
  display: flex;
  align-items: center;
}

.plus-delete-icon {
  margin-right: 8px;
}

.upload-button:hover {
  background-color: #F6F6F6;
}

.text-container {
  flex-grow: 1;
}

.header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 10px;
}

.initial-prompt {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.spinner-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
}

.spinner {
  border: 8px solid #F3F3F3;
  border-top: 8px solid #5AA1AF;
  border-radius: 50%;
  width: 60px;
  height: 60px;
  animation: spin 2s linear infinite;
  margin-bottom: 20px;
}

@keyframes spin {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

::-webkit-scrollbar {
  width: 10px;
}

::-webkit-scrollbar-thumb {
  background-color: #E3F1F8;
  border-radius: 5px;
  cursor: pointer;
}
</style>
