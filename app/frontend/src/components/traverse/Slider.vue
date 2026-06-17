<script setup>
import { computed } from 'vue'

defineProps({
  active: {
    type: Boolean,
    required: true,
  },
})

const beta = defineModel('beta', {
  type: Number,
  required: true,
})

const clampedBeta = computed(() => {
  const value = Number(beta.value)
  if (!Number.isFinite(value)) return 0
  return Math.max(0, Math.min(1, value))
})

const fillStyle = computed(() => ({
  position: 'absolute',
  left: 0,
  top: 0,
  height: '100%',
  width: `${clampedBeta.value * 100}%`,
  background: '#48c78e',
  borderRadius: '4px',
}))

const onInput = (event) => {
  beta.value = Number.parseFloat(event.target.value)
}
</script>

<template>
  <section class="navigation-controls">
    <div class="slider-header">
      <span class="header-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="18" height="18">
          <line x1="2" y1="4" x2="22" y2="8" stroke="currentColor" stroke-width="2" />
          <path d="M12,6 L12,14" stroke="currentColor" stroke-width="2" fill="none" />
          <path d="M9,14 L15,14 A1,1 0 0 1 16,15 L16,17 A2,2 0 0 1 14,19 L10,19 A2,2 0 0 1 8,17 L8,15 A1,1 0 0 1 9,14" stroke="currentColor" stroke-width="2" fill="none" />
        </svg>
      </span>
      <h3 class="header-title">Traverse</h3>
    </div>
    <div class="field">
      <div class="slider-wrapper">
        <div class="value-row">
          <span class="value-label">0.0</span>
          <span class="value-label">{{ clampedBeta.toFixed(2) }}</span>
          <span class="value-label">1.0</span>
        </div>

        <div class="progress-bar-container" style="position: relative; height: 1rem;">
          <div class="segments-wrapper" style="width: 100%; height: 100%; background: #e8e8e8; border-radius: 4px; position: relative;">
            <div style="position: absolute; left: 0; top: 0; height: 100%; width: 100%; background: #dbdbdb; border-radius: 4px;"></div>
            <div :style="fillStyle"></div>
          </div>

          <input
            class="beta-slider-overlay"
            type="range"
            min="0"
            max="1"
            step="0.001"
            :disabled="!active"
            :value="clampedBeta"
            @input="onInput"
            style="left: 0; width: 100%;"
          >
        </div>
      </div>
    </div>
  </section>
</template>

<style scoped>
.slider-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.header-icon {
  display: inline-flex;
  color: #1f2937;
}

.header-title {
  margin: 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: #1f2937;
}

.slider-wrapper {
  position: relative;
  width: 100%;
  padding: 4px 8px 8px;
}

.value-row {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.value-label {
  font-size: 0.75rem;
  font-weight: 700;
}

.progress-bar-container {
  position: relative;
  height: 1rem;
  background: repeating-linear-gradient(
    45deg,
    #e8e8e8,
    #e8e8e8 6px,
    #dedede 6px,
    #dedede 12px
  );
  border-radius: 6px;
  display: flex;
  align-items: center;
}

.segments-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: 6px;
  overflow: hidden;
}

.beta-slider-overlay {
  position: absolute;
  top: 0;
  margin: 0;
  height: 100%;
  -webkit-appearance: none;
  background: transparent;
  cursor: pointer;
  z-index: 5;
}

.beta-slider-overlay::-webkit-slider-runnable-track {
  background: transparent;
}

.beta-slider-overlay::-moz-range-track {
  background: transparent;
}

.beta-slider-overlay::-webkit-slider-thumb {
  -webkit-appearance: none;
  height: 1.4rem;
  width: 16px;
  background: #ffffff;
  border: 2px solid #485fc7;
  border-radius: 4px;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
}

.beta-slider-overlay::-moz-range-thumb {
  height: 1.4rem;
  width: 16px;
  background: #ffffff;
  border: 2px solid #485fc7;
  border-radius: 4px;
  cursor: pointer;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
}

.beta-slider-overlay:disabled {
  cursor: default;
}

.beta-slider-overlay:disabled::-webkit-slider-thumb {
  cursor: default;
}

.beta-slider-overlay:disabled::-moz-range-thumb {
  cursor: default;
}

.beta-slider-overlay:focus {
  outline: none;
}
</style>
