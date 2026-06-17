<script setup>
import { ref, computed, watch } from 'vue'
import logoMGA from '../assets/mga_logo.svg'

const isOpen = defineModel('open', { type: Boolean, default: true })
const slideIndex = ref(0)

watch(isOpen, (open) => {
  if (open) slideIndex.value = 0
})

const slides = [
  {
    title: 'Welcome to MGA Compass',
    text: 'Near-optimal solutions cost a small percentage more than the optimum. While slightly expensive, they can be preferable for political feasibility, social acceptance, or environmental impact. This platform helps you explore them in real time.',
    icon: 'logo',
  },
  {
    title: 'Set your priorities',
    textParts: [
      'Each row in the Navigation panel is a property of interest. Drag rows by their ',
      { icon: 'drag' },
      ' icon to reorder them by priority, then set a desired direction (',
      { icon: 'increase' },
      ', ',
      { icon: 'decrease' },
      ', ',
      { icon: 'equal' },
      ') and delta for each one.',
    ],
    icon: 'compass',
  },
  {
    title: 'Explore & Reset',
    text: 'Click "Explore" to search for a new solution that respects your constraints. "Reset" reverts your changes back to the last explored point.',
    icon: 'explore',
  },
  {
    title: 'Absolute vs. percentage view',
    text: 'Use the toggle in the Navigation header to switch between absolute values and percentage-of-minimum values.',
    icon: 'percent',
  },
  {
    title: 'Traverse the path',
    text: 'Switch to Traverse mode to slide between your current point and the point found through navigation, and see how the solution changes along the way.',
    icon: 'chairlift',
  },
  {
    title: "You're all set!",
    text: "You're ready to start exploring! Find the source code and documentation via the GitHub icon in the top right, and reopen this guide anytime using the lightbulb icon.",
    icon: 'check',
  },
]

const currentSlide = computed(() => slides[slideIndex.value])
const isFirst = computed(() => slideIndex.value === 0)
const isLast = computed(() => slideIndex.value === slides.length - 1)

const close = () => {
  isOpen.value = false
}

const next = () => {
  if (isLast.value) return
  slideIndex.value++
}

const prev = () => {
  if (isFirst.value) return
  slideIndex.value--
}

const goTo = (index) => {
  slideIndex.value = index
}
</script>

<template>
  <div v-if="isOpen" class="tutorial-overlay" @click.self="close">
    <div class="tutorial-modal">
      <button class="close-btn" type="button" aria-label="Close tutorial" @click="close">
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="4" y1="4" x2="20" y2="20" />
          <line x1="20" y1="4" x2="4" y2="20" />
        </svg>
      </button>

      <div class="tutorial-icon" aria-hidden="true">
        <img v-if="currentSlide.icon === 'logo'" :src="logoMGA" alt="" class="icon-logo" />
        <svg v-else-if="currentSlide.icon === 'compass'" viewBox="0 0 24 24" width="48" height="48">
          <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M17,7L13.5,13.5L7,17L10.5,10.5L17,7M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11Z" />
        </svg>
        <svg v-else-if="currentSlide.icon === 'explore'" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="7" />
          <line x1="21" y1="21" x2="16.65" y2="16.65" />
        </svg>
        <svg v-else-if="currentSlide.icon === 'percent'" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="7" cy="7" r="2.5" />
          <circle cx="17" cy="17" r="2.5" />
          <line x1="19" y1="5" x2="5" y2="19" />
        </svg>
        <svg v-else-if="currentSlide.icon === 'chairlift'" viewBox="0 0 24 24" width="48" height="48">
          <line x1="2" y1="4" x2="22" y2="8" stroke="currentColor" stroke-width="1.5" />
          <path d="M12,6 L12,14" stroke="currentColor" stroke-width="1.5" fill="none" />
          <path d="M9,14 L15,14 A1,1 0 0 1 16,15 L16,17 A2,2 0 0 1 14,19 L10,19 A2,2 0 0 1 8,17 L8,15 A1,1 0 0 1 9,14" stroke="currentColor" stroke-width="1.5" fill="none" />
        </svg>
        <svg v-else-if="currentSlide.icon === 'check'" viewBox="0 0 24 24" width="48" height="48" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M8 12.5l2.5 2.5L16 9" stroke-linecap="round" stroke-linejoin="round" />
        </svg>
      </div>

      <h2 class="tutorial-title">{{ currentSlide.title }}</h2>
      <p class="tutorial-text">
        <template v-if="currentSlide.textParts">
          <template v-for="(part, idx) in currentSlide.textParts" :key="idx">
            <span v-if="typeof part === 'string'">{{ part }}</span>
            <span v-else-if="part.icon === 'drag'" class="inline-icon inline-icon-plain" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="16" height="16">
                <path fill="currentColor" d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
              </svg>
            </span>
            <span v-else-if="part.icon === 'increase'" class="inline-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="14" height="14">
                <path fill="currentColor" d="M7 14l5-5 5 5H7z" />
              </svg>
            </span>
            <span v-else-if="part.icon === 'equal'" class="inline-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="14" height="14">
                <path fill="currentColor" d="M19 9.5H5v-2h14v2zM19 16.5H5v-2h14v2z" />
              </svg>
            </span>
            <span v-else-if="part.icon === 'decrease'" class="inline-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="14" height="14">
                <path fill="currentColor" d="M7 10l5 5 5-5H7z" />
              </svg>
            </span>
          </template>
        </template>
        <template v-else>{{ currentSlide.text }}</template>
      </p>

      <div class="tutorial-dots">
        <button
          v-for="(slide, index) in slides"
          :key="index"
          type="button"
          class="dot"
          :class="{ 'is-active': index === slideIndex }"
          :aria-label="`Go to slide ${index + 1}`"
          @click="goTo(index)"
        />
      </div>

      <div class="tutorial-actions">
        <button type="button" class="action-btn secondary" :disabled="isFirst" @click="prev" aria-label="Previous slide">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="15 18 9 12 15 6" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <button type="button" class="action-btn primary" :disabled="isLast" @click="next" aria-label="Next slide">
          <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="9 18 15 12 9 6" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<style scoped>
.tutorial-overlay {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.tutorial-modal {
  position: relative;
  background: #ffffff;
  border-radius: 8px;
  padding: 32px;
  width: 90%;
  max-width: 420px;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.25);
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}

.close-btn {
  position: absolute;
  top: 12px;
  right: 12px;
  border: none;
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
  display: inline-flex;
  padding: 4px;
  border-radius: 4px;
  transition: color 0.2s, background 0.2s;
}

.close-btn:hover {
  color: #1f2937;
  background: #f0f0f0;
}

.tutorial-icon {
  display: inline-flex;
  color: #3e8ed0;
  margin-bottom: 16px;
}

.icon-logo {
  height: 48px;
}

.tutorial-title {
  margin: 0 0 8px;
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
}

.tutorial-text {
  margin: 0 0 20px;
  color: #4b5563;
  line-height: 1.5;
  min-height: 6em;
}

.inline-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  background: #f9fafb;
  color: #6b7280;
  vertical-align: middle;
  margin: 0 2px;
}

.inline-icon-plain {
  width: auto;
  height: auto;
  border: none;
  background: transparent;
  color: #9ca3af;
}

.tutorial-dots {
  display: flex;
  gap: 8px;
  margin-bottom: 20px;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  border: none;
  padding: 0;
  background: #e3e3e3;
  cursor: pointer;
  transition: background 0.2s;
}

.dot.is-active {
  background: #3e8ed0;
}

.tutorial-actions {
  display: flex;
  gap: 8px;
  width: 100%;
  justify-content: center;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid transparent;
  border-radius: 50%;
  width: 36px;
  height: 36px;
  cursor: pointer;
  transition: background 0.2s, color 0.2s;
}

.action-btn.primary {
  background: #3e8ed0;
  color: #ffffff;
}

.action-btn.primary:not(:disabled):hover {
  background: #3577ad;
}

.action-btn.secondary {
  background: #ffffff;
  color: #3e8ed0;
  border-color: #3e8ed0;
}

.action-btn.secondary:not(:disabled):hover {
  background: #eff6ff;
}

.action-btn:disabled {
  opacity: 0.4;
  cursor: default;
}
</style>
