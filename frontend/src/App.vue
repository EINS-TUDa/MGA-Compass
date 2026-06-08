<script setup>
import { ref } from 'vue'
import Navigate from './components/Navigate.vue'
import Traverse from './components/Traverse.vue'
import logoTU from './assets/logo_tu_darmstadt.svg'
import logoMGA from './assets/mga_logo.svg'

const beta = ref(0)

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
    <header class="main-header">
      <div class="brand-left">
        <img :src="logoMGA" alt="MGA Logo" class="logo-mga" />
        <h1 class="dashboard-title">MGA Exploration Dashboard</h1>
      </div>

      <div class="header-center">
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
        <img :src="logoTU" alt="TU Darmstadt" class="logo-tu" />
      </div>
    </header>

    <main class="content-area">
      <Navigate :active="mode === 'naviagte'" :beta="beta" v-model:path="path" @navigated="onNavigated" />
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

.dashboard-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #000000;
}

.header-center {
  position: absolute;
  top: 50%;
  left: 50%;
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
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
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
  gap: 10px;
  overflow: hidden;
  padding: 10px;
}

.content-area > * {
  min-width: 0;
}
</style>
