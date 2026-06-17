<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import { formatPercent, formatCompact } from '../../shared/numberFormat'

const props = defineProps({
  name: {
    type: String,
    required: true,
  },
  order: {
    type: Number,
    required: true,
  },
  min: {
    type: Number,
    required: true,
  },
  max: {
    type: Number,
    required: true,
  },
  start: {
    type: Number,
    required: true,
  },
  end: {
    type: Number,
    required: true,
  },
  info: {
    type: String,
    required: true,
  },
  unit: {
    type: String,
    default: '',
  },
  value: {
    type: Number,
    required: true,
  },
  disabled: {
    type: Boolean,
    default: false,
  },
  normalize: {
    type: Boolean,
    default: false,
  },
  objective: {
    type: Boolean,
    default: false,
  },
})

const deltaModel = defineModel('delta', {
  type: Number,
})

const directionModel = defineModel('direction', {
  type: String,
})

const selectDirection = (direction) => {
  if (props.disabled) return
  directionModel.value = directionModel.value === direction ? '' : direction
}

defineEmits(['handle-mousedown'])

const showInfo = ref(false)
const infoBtnRef = ref(null)
const tooltipRef = ref(null)
const tooltipStyle = ref({})

const onInfoClick = () => {
  showInfo.value = !showInfo.value
}

const onDocClick = (e) => {
  if (infoBtnRef.value && !infoBtnRef.value.contains(e.target)) {
    showInfo.value = false
  }
}

const onScroll = () => {
  showInfo.value = false
}

const TOOLTIP_WIDTH = 220
const PAD = 8

watch(showInfo, async (val) => {
  if (val) {
    document.addEventListener('click', onDocClick)
    document.addEventListener('scroll', onScroll, true)
    await nextTick()
    if (!infoBtnRef.value) return
    const anchor = infoBtnRef.value.getBoundingClientRect()
    let left = anchor.left + anchor.width / 2 - TOOLTIP_WIDTH / 2
    left = Math.max(PAD, Math.min(left, window.innerWidth - TOOLTIP_WIDTH - PAD))
    tooltipStyle.value = {
      position: 'fixed',
      top: `${anchor.bottom + 6}px`,
      left: `${left}px`,
      width: `${TOOLTIP_WIDTH}px`,
    }
  } else {
    document.removeEventListener('click', onDocClick)
    document.removeEventListener('scroll', onScroll, true)
    tooltipStyle.value = {}
  }
})

onUnmounted(() => {
  document.removeEventListener('click', onDocClick)
  document.removeEventListener('scroll', onScroll, true)
})

const isDeltaDisabled = computed(() => props.disabled || directionModel.value === '')

const toDisplay = (v) => (props.normalize && props.min !== 0) ? (Number(v) / props.min) * 100 : Number(v)
const fromDisplay = (v) => (props.normalize && props.min !== 0) ? (Number(v) / 100) * props.min : Number(v)

const displayDecimals = computed(() => {
  const range = Math.abs(toDisplay(props.max) - toDisplay(props.min))
  const step = range * 0.001
  if (!Number.isFinite(step) || step <= 0) return 0
  return Math.min(12, Math.max(0, Math.ceil(-Math.log10(step))))
})

const formatByRange = (value) => {
  const num = toDisplay(value)
  if (!Number.isFinite(num)) return String(value)

  if (props.normalize) {
    return formatPercent(num)
  }

  const displayMin = toDisplay(props.min)
  const displayMax = toDisplay(props.max)
  const minAbs = Math.min(Math.abs(displayMin), Math.abs(displayMax))
  const maxAbs = Math.max(Math.abs(displayMin), Math.abs(displayMax))

  if (minAbs >= 1_000) {
    return formatCompact(num, maxAbs)
  }

  const range = Math.abs(displayMax - displayMin)
  const step = range * 0.001
  if (!Number.isFinite(step) || step <= 0) return String(num)

  const rounded = Math.round(num / step) * step
  return rounded.toFixed(displayDecimals.value).replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
}

const compactDivisor = computed(() => {
  const minAbs = Math.min(Math.abs(toDisplay(props.min)), Math.abs(toDisplay(props.max)))
  const maxAbs = Math.max(Math.abs(toDisplay(props.min)), Math.abs(toDisplay(props.max)))
  if (minAbs >= 1_000) {
    if (maxAbs >= 1_000_000_000) return 1_000_000_000
    if (maxAbs >= 1_000_000) return 1_000_000
    return 1_000
  }
  return 1
})

const displayDelta = computed({
  get: () => toDisplay(deltaModel.value ?? 0) / compactDivisor.value,
  set: (v) => { deltaModel.value = fromDisplay(v * compactDivisor.value) },
})

const rangeMeta = computed(() => {
  const min = Number(props.min)
  const max = Number(props.max)
  const current = Number(props.value)
  const start = Number(props.start)
  const end = Number(props.end)
  const delta = Math.max(0, Number(deltaModel.value) || 0)
  const total = max - min

  if (!Number.isFinite(total) || total <= 0) {
    return {
      highlight: { left: '0%', width: '0%' },
      fixed: { width: '0%' },
      active: { left: '0%', width: '0%' },
      indicators: [],
    }
  }

  const rStart = Math.min(start, end)
  const rEnd = Math.max(start, end)

  const indicatorBase = props.disabled ? start : current
  const points = []
  if (directionModel.value === '>=') {
    points.push(indicatorBase + Math.min(delta, max - indicatorBase))
  }
  if (directionModel.value === '<=') {
    points.push(indicatorBase - Math.min(delta, indicatorBase - min))
  }
  if (directionModel.value === '==') {
    const boundedDelta = Math.min(delta, max - indicatorBase, indicatorBase - min)
    points.push(indicatorBase + boundedDelta)
    points.push(indicatorBase - boundedDelta)
  }

  return {
    highlight: {
      left: `${Math.max(0, Math.min(100, ((rStart - min) / total) * 100))}%`,
      width: `${Math.max(0, Math.min(100, ((rEnd - rStart) / total) * 100))}%`,
    },
    fixed: {
      width: `${Math.max(0, Math.min(100, ((Math.min(current, rStart) - min) / total) * 100))}%`,
    },
    active: {
      left: `${Math.max(0, Math.min(100, ((rStart - min) / total) * 100))}%`,
      width: `${Math.max(0, Math.min(100, ((Math.max(0, current - rStart)) / total) * 100))}%`,
    },
    indicators: [...new Set(points)].map((point) => {
      let dir = 'right'

      if (directionModel.value === '<=') {
        dir = 'left'
      } else if (directionModel.value === '>=') {
        dir = 'right'
      } else if (directionModel.value === '==') {
        dir = point > indicatorBase ? 'left' : 'right'
      }

      return {
        left: `${Math.max(0, Math.min(100, ((point - min) / total) * 100))}%`,
        dir,
      }
    }),
  }
})

const normalizeDeltaOnBlur = () => {
  if (!Number.isFinite(Number(deltaModel.value))) deltaModel.value = 0
}

const fmtDelta = (v) => String(parseFloat(Number(v).toFixed(2)))

const deltaInputStr = ref(fmtDelta(displayDelta.value))
let deltaInputFocused = false

watch(displayDelta, (v) => {
  if (!deltaInputFocused) deltaInputStr.value = fmtDelta(v)
})

const onDeltaFocus = () => { deltaInputFocused = true }

const onDeltaInput = (e) => {
  deltaInputStr.value = e.target.value
  const v = parseFloat(String(e.target.value).replace(',', '.'))
  if (Number.isFinite(v) && v >= 0) displayDelta.value = v
}

const onDeltaBlur = () => {
  deltaInputFocused = false
  normalizeDeltaOnBlur()
  deltaInputStr.value = fmtDelta(displayDelta.value)
}

const progressRef = ref(null)
let dragging = false

const onIndicatorMouseDown = (event) => {
  if (isDeltaDisabled.value) return
  dragging = true
  event.preventDefault()
}

const onMouseMove = (event) => {
  if (!dragging || !progressRef.value) return
  const rect = progressRef.value.getBoundingClientRect()
  const pct = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
  const draggedValue = props.min + pct * (props.max - props.min)
  const base = props.disabled ? props.start : props.value
  if (directionModel.value === '>=') deltaModel.value = Math.max(0, draggedValue - base)
  else if (directionModel.value === '<=') deltaModel.value = Math.max(0, base - draggedValue)
  else if (directionModel.value === '==') deltaModel.value = Math.abs(draggedValue - base)
}

const onMouseUp = () => { dragging = false }

onMounted(() => {
  document.addEventListener('mousemove', onMouseMove)
  document.addEventListener('mouseup', onMouseUp)
})

onUnmounted(() => {
  document.removeEventListener('mousemove', onMouseMove)
  document.removeEventListener('mouseup', onMouseUp)
})
</script>

<template>
  <section class="priority-card" :class="{ 'is-disabled': props.disabled, 'is-objective': props.objective }">
    <div class="priority-header">
      <div class="priority-title-wrap">
        <span class="drag-handle" aria-hidden="true" title="Drag to reorder" @mousedown="$emit('handle-mousedown')">
          <svg viewBox="0 0 24 24" width="18" height="18">
            <path fill="currentColor" d="M11 18c0 1.1-.9 2-2 2s-2-.9-2-2 .9-2 2-2 2 .9 2 2zm-2-8c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0-6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm6 4c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z" />
          </svg>
        </span>
        <span class="priority-index">{{ order + 1 }}.</span>
        <h4 class="priority-title">{{ name }}<span v-if="unit" class="name-unit"> ({{ unit }})</span><span v-if="objective" class="obj-marker"> ∗</span></h4>
        <span v-if="info" ref="infoBtnRef" class="info-wrap">
          <button
            type="button"
            class="info-btn"
            :class="{ 'is-active': showInfo }"
            @click="onInfoClick"
            aria-label="Show info"
          >
            <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true">
              <circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" stroke-width="2"/>
              <path fill="currentColor" d="M12 10.5a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm-1 1.5h2v5h-2z"/>
            </svg>
          </button>
          <Teleport to="body">
            <div
              v-if="showInfo"
              ref="tooltipRef"
              class="info-tooltip"
              :style="tooltipStyle"
            >
              <span v-if="unit" class="tooltip-unit">{{ unit }}</span>
              {{ info }}
            </div>
          </Teleport>
        </span>
      </div>
      <div class="direction-group">
        <button
          type="button"
          class="direction-btn"
          :class="{ 'is-active': directionModel === '>=' }"
          @click="selectDirection('>=')"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
            <path fill="currentColor" d="M7 14l5-5 5 5H7z" />
          </svg>
        </button>
        <button
          type="button"
          class="direction-btn"
          :style="order === 0 ? 'visibility: hidden;' : ''"
          :class="{ 'is-active': directionModel === '==' }"
          @click="selectDirection('==')"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
            <path fill="currentColor" d="M19 9.5H5v-2h14v2zM19 16.5H5v-2h14v2z" />
          </svg>
        </button>
        <button
          type="button"
          class="direction-btn"
          :class="{ 'is-active': directionModel === '<=' }"
          @click="selectDirection('<=')"
        >
          <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
            <path fill="currentColor" d="M7 10l5 5 5-5H7z" />
          </svg>
        </button>
      </div>
    </div>


<div class="range-values">
      <span>{{ formatByRange(min) }}</span>
      <span class="current-value">{{ formatByRange(value) }}</span>
      <span>{{ formatByRange(max) }}</span>
    </div>

    <div ref="progressRef" class="progress-container">
      <div class="segments-wrapper">
        <div class="range-highlight" :style="rangeMeta.highlight"></div>
        <div class="segment-fixed" :style="rangeMeta.fixed"></div>
        <div class="segment-active" :style="rangeMeta.active"></div>
      </div>
      <div
        v-for="(ind, idx) in rangeMeta.indicators"
        :key="idx"
        class="target-indicator"
        :class="ind.dir"
        :style="{ left: ind.left, cursor: isDeltaDisabled ? 'default' : 'ew-resize' }"
        @mousedown="onIndicatorMouseDown"
      ></div>
    </div>

    <div class="delta-row">
      <label class="delta-label" for="delta-input">Delta</label>
      <input
        id="delta-input"
        type="text"
        inputmode="decimal"
        :value="deltaInputStr"
        :disabled="isDeltaDisabled"
        class="delta-input"
        @focus="onDeltaFocus"
        @input="onDeltaInput"
        @blur="onDeltaBlur"
      >
    </div>
  </section>
</template>

<style scoped>
.priority-card {
  border: 1px solid #dbdbdb;
  border-radius: 4px;
  background: #ffffff;
  padding: 12px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}


.priority-card.is-disabled {
  background: #f2f2f2;
  pointer-events: none;
}

.priority-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 12px;
}

.priority-title-wrap {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 0;
}

.drag-handle {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  color: #9ca3af;
  cursor: grab;
}

.priority-index {
  font-size: 0.95rem;
  font-weight: 600;
  color: #4b5563;
}

.priority-title {
  margin: 0;
  font-size: 1rem;
  font-weight: 600;
  color: #1f2937;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.name-unit {
  font-size: 0.8rem;
  font-weight: 400;
  color: #6b7280;
}

.obj-marker {
  font-size: 1.3rem;
  line-height: 1;
}

.info-wrap {
  position: relative;
  display: inline-flex;
  align-items: center;
}

.info-btn {
  flex: none;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 20px;
  height: 20px;
  border: none;
  border-radius: 50%;
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
  padding: 0;
  transition: color 0.15s, background 0.15s;
}

.info-btn:hover,
.info-btn.is-active {
  color: #3e8ed0;
  background: #eff6ff;
}

.tooltip-unit {
  display: inline-block;
  font-size: 0.7rem;
  font-weight: 600;
  color: #3e8ed0;
  background: #eff6ff;
  border-radius: 3px;
  padding: 1px 5px;
  margin-right: 5px;
  vertical-align: middle;
}

.info-tooltip {
  background: #ffffff;
  color: #1f2937;
  font-size: 0.8rem;
  line-height: 1.5;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
  z-index: 9999;
  pointer-events: none;
}

.direction-group {
  display: flex;
  gap: 4px;
}

.direction-btn {
  width: 28px;
  height: 28px;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  background: #f9fafb;
  color: #6b7280;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}

.direction-btn.is-active {
  border-color: #3e8ed0;
  background: #3e8ed0;
  color: #ffffff;
}


.range-values {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #6b7280;
  margin-bottom: 6px;
}

.current-value {
  color: #111827;
  font-weight: 700;
}

.progress-container {
  position: relative;
  height: 0.75rem;
  background: repeating-linear-gradient(
    45deg,
    #e8e8e8,
    #e8e8e8 6px,
    #dedede 6px,
    #dedede 12px
  );
  border-radius: 4px;
  display: flex;
  align-items: center;
}

.segments-wrapper {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border-radius: 4px;
  overflow: hidden;
}

.range-highlight {
  position: absolute;
  height: 100%;
  background-color: #f5f5f5;
  z-index: 1;
}

.segment-fixed {
  position: absolute;
  height: 100%;
  background-color: #485fc7;
  z-index: 2;
}

.segment-active {
  position: absolute;
  height: 100%;
  background-color: #485fc7;
  z-index: 2;
}

.is-disabled .segment-active {
  background-color: #48c78e;
}

.target-indicator {
  position: absolute;
  top: -8px;
  width: 2px;
  height: calc(100% + 16px);
  background-color: #ffdd57;
  z-index: 3;
}

.target-indicator::after,
.target-indicator::before {
  content: '';
  position: absolute;
  width: 0;
  height: 0;
  border-top: 4px solid transparent;
  border-bottom: 4px solid transparent;
}

.target-indicator::after {
  top: 0;
}

.target-indicator::before {
  bottom: 0;
}

.target-indicator.right::after,
.target-indicator.right::before {
  left: 2px;
  border-left: 6px solid #ffdd57;
}

.target-indicator.left::after,
.target-indicator.left::before {
  right: 2px;
  border-right: 6px solid #ffdd57;
}

.delta-row {
  margin-top: 10px;
  display: inline-flex;
  align-items: center;
  border: 1px solid #d1d5db;
  border-radius: 4px;
  overflow: hidden;
}

.delta-label {
  padding: 6px 8px;
  font-size: 0.8rem;
  color: #4b5563;
  background: #f3f4f6;
}

.delta-input {
  width: 72px;
  border: 0;
  padding: 6px 8px;
  text-align: center;
  outline: none;
}

.delta-input:disabled {
  background: #f9fafb;
  color: #9ca3af;
}

</style>
