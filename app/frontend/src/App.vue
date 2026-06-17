<script setup>
import { ref, onUnmounted, watch } from 'vue'
import Navigate from './components/Navigate.vue'
import Traverse from './components/Traverse.vue'
import TutorialModal from './components/TutorialModal.vue'
import logoTU from './assets/logo_tu_darmstadt.svg'
import logoMGA from './assets/mga_logo.svg'

const TUTORIAL_SEEN_KEY = 'mga-tutorial-seen'
const showTutorial = ref(localStorage.getItem(TUTORIAL_SEEN_KEY) !== 'true')

watch(showTutorial, (open) => {
  if (!open) localStorage.setItem(TUTORIAL_SEEN_KEY, 'true')
})

const openGithub = () => {
  window.open('https://github.com/EINS-TUDa/MGA-Compass', '_blank', 'noopener,noreferrer')
}
const beta = ref(0)
const splitPercent = ref(40)
const contentAreaRef = ref(null)
let dragging = false

const onDividerMousedown = (e) => {
  dragging = true
  e.preventDefault()
  document.addEventListener('mousemove', onMousemove)
  document.addEventListener('mouseup', onMouseup)
}

const onMousemove = (e) => {
  if (!dragging) return
  const rect = contentAreaRef.value.getBoundingClientRect()
  const pct = ((e.clientX - rect.left) / rect.width) * 100
  splitPercent.value = Math.min(Math.max(pct, 15), 85)
}

const onMouseup = () => {
  dragging = false
  document.removeEventListener('mousemove', onMousemove)
  document.removeEventListener('mouseup', onMouseup)
}

onUnmounted(() => {
  document.removeEventListener('mousemove', onMousemove)
  document.removeEventListener('mouseup', onMouseup)
})

/** @type {import('vue').Ref<'naviagte' | 'traverse'>} */
const mode = ref('naviagte')

/**
 * @typedef {Object} PathItem
 * @property {number} beta
 * @property {Map<number, number>} alpha
 */

/** @type {import('vue').Ref<PathItem[]>} */
const path = ref([])

const onNavigated = () => {
  mode.value = 'traverse'
  beta.value = 0
}

const toggleMode = () => {
  mode.value = mode.value === 'naviagte' ? 'traverse' : 'naviagte'
}
</script>

<template>
  <div class="full-window">
    <TutorialModal v-model:open="showTutorial" />

    <header class="main-header">
      <div class="brand-left">
        <img :src="logoMGA" alt="MGA Logo" class="logo-mga" />
        <h1 class="dashboard-title">MGA Compass</h1>
      </div>

      <div class="header-center" :style="{ left: `calc(${splitPercent}vw - ${splitPercent * 0.2}px + 14px)` }">
        <button
          type="button"
          class="mode-toggle"
          :class="{ 'is-traverse': mode === 'traverse' }"
          @click="toggleMode"
          aria-label="Toggle between navigate and traverse"
        >
          <span class="toggle-track">
            <span class="compass-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="16" height="16">
                <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M17,7L13.5,13.5L7,17L10.5,10.5L17,7M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11Z" />
              </svg>
            </span>
            <span class="chairlift-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="18" height="18">
                <line x1="2" y1="4" x2="22" y2="8" stroke="currentColor" stroke-width="1.5" />
                <path d="M12,6 L12,14" stroke="currentColor" stroke-width="1.5" fill="none" />
                <path d="M9,14 L15,14 A1,1 0 0 1 16,15 L16,17 A2,2 0 0 1 14,19 L10,19 A2,2 0 0 1 8,17 L8,15 A1,1 0 0 1 9,14" stroke="currentColor" stroke-width="1.5" fill="none" />
              </svg>
            </span>
          </span>
          <span class="toggle-thumb" aria-hidden="true">
            <svg v-if="mode === 'naviagte'" viewBox="0 0 24 24" width="16" height="16">
              <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M17,7L13.5,13.5L7,17L10.5,10.5L17,7M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11Z" />
            </svg>
            <svg v-else viewBox="0 0 24 24" width="18" height="18">
              <line x1="2" y1="4" x2="22" y2="8" stroke="currentColor" stroke-width="1.5" />
              <path d="M12,6 L12,14" stroke="currentColor" stroke-width="1.5" fill="none" />
              <path d="M9,14 L15,14 A1,1 0 0 1 16,15 L16,17 A2,2 0 0 1 14,19 L10,19 A2,2 0 0 1 8,17 L8,15 A1,1 0 0 1 9,14" stroke="currentColor" stroke-width="1.5" fill="none" />
            </svg>
          </span>
        </button>
      </div>

      <div class="brand-right">
        <button
          type="button"
          class="help-btn"
          @click="showTutorial = true"
          aria-label="Open tutorial"
        >
          <svg viewBox="0 0 24 24" width="24" height="24">
            <circle cx="12" cy="12" r="11" fill="none" stroke="currentColor" stroke-width="1.5" />
            <path fill="currentColor" transform="translate(12 12) scale(0.65) translate(-12 -12)" d="M12,2A7,7 0 0,0 5,9C5,11.38 6.19,13.47 8,14.74V17A1,1 0 0,0 9,18H15A1,1 0 0,0 16,17V14.74C17.81,13.47 19,11.38 19,9A7,7 0 0,0 12,2M9,21A1,1 0 0,0 10,22H14A1,1 0 0,0 15,21V20H9V21Z" />
          </svg>
        </button>
        <a
          class="github-link"
          href="https://github.com/EINS-TUDa/MGA-Compass"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="GitHub repository"
          @click.prevent="openGithub"
        >
          <svg viewBox="0 0 16 16" width="24" height="24" fill="currentColor">
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
        </a>
        <img :src="logoTU" alt="TU Darmstadt" class="logo-tu" />
      </div>
    </header>

    <main class="content-area" ref="contentAreaRef">
      <Navigate
        :active="mode === 'naviagte'"
        :beta="beta"
        v-model:path="path"
        @navigated="onNavigated"
        :style="{ flex: 'none', width: splitPercent + '%' }"
      />
      <div class="divider" @mousedown="onDividerMousedown">
        <span class="divider-handle" />
      </div>
      <Traverse :active="mode === 'traverse'" :path="path" v-model:beta="beta" />
    </main>
  </div>
</template>

<style>
html,
body {
  margin: 0;
  padding: 0;
  height: 100vh;
  overflow: hidden;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

#app {
  height: 100%;
}

.full-window {
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
  background-color: #f5f5f5;
}

.main-header {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 16px;
  background: #ffffff;
  border-bottom: 1px solid #dddddd;
}

.brand-left,
.brand-right {
  display: flex;
  align-items: center;
}

.logo-mga {
  height: 32px;
  margin-right: 12px;
}

.logo-tu {
  height: 30px;
}

.help-btn {
  display: inline-flex;
  align-items: center;
  margin-right: 8px;
  border: none;
  background: transparent;
  padding: 0;
  color: #1f2937;
  cursor: pointer;
  transition: color 0.2s;
}

.help-btn:hover {
  color: #3e8ed0;
}

.github-link {
  display: inline-flex;
  align-items: center;
  margin-right: 8px;
  color: #1f2937;
  transition: color 0.2s;
}

.github-link:hover {
  color: #3e8ed0;
}

.dashboard-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 700;
  color: #1f2937;
}

.header-center {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
}

.mode-toggle {
  width: 70px;
  height: 32px;
  padding: 2px;
  border: none;
  border-radius: 16px;
  background-color: #f0f0f0;
  position: relative;
  cursor: pointer;
  transition: background-color 0.3s;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.15);
}

.toggle-track {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  padding: 0 8px;
  color: #888888;
}

.toggle-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border-radius: 50%;
  background-color: #ffffff;
  color: #3e8ed0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.mode-toggle.is-traverse .toggle-thumb {
  transform: translateX(38px);
}

.compass-icon,
.chairlift-icon {
  display: inline-flex;
  transition: opacity 0.3s;
  opacity: 0.3;
}

.mode-toggle.is-traverse .chairlift-icon {
  opacity: 1;
}

.mode-toggle:not(.is-traverse) .compass-icon {
  opacity: 1;
}

.content-area {
  display: flex;
  flex: 1;
  overflow: hidden;
  padding: 10px;
  gap: 0;
}

.content-area > * {
  min-width: 0;
}

.divider {
  flex: none;
  width: 8px;
  cursor: col-resize;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  user-select: none;
}

.divider:hover .divider-handle,
.divider:active .divider-handle {
  background: #3e8ed0;
}

.divider-handle {
  width: 3px;
  height: 40px;
  border-radius: 2px;
  background: #cccccc;
  transition: background 0.2s;
}
</style>
