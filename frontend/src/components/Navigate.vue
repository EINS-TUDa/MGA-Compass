<script setup>
import { computed, onMounted, ref, watch, watchEffect } from 'vue'
import { get, post } from '@/shared/api'
import MessageBox from './navigate/MessageBox.vue'
import PriorityItem from './navigate/PriorityItem.vue'

const props = defineProps({
  active: {
    type: Boolean,
    required: true,
  },
  beta: {
    type: Number,
    required: true,
  },
})

const path = defineModel('path')

/**
 * @typedef {Object} Priority
 * @property {string} name
 * @property {string} info
 * @property {number} min
 * @property {number} max
 * @property {number} order
 * @property {number} start
 * @property {number} end
 * @property {number} value
 * @property {number} delta
 * @property {'' | '>=' | '<=' | '=='} direction
 */

/** @type {import('vue').Ref<Priority[]>} */
const priorities = ref([])


// const DEBUG_FALLBACK_PRIORITIES = reactive([
//   { name: 'Cost', info: 'Debug priority for UI preview', min: 0, max: 100, order: 0, start: 65, end: 30, value: 45, delta: 10, direction: '>=' },
//   { name: 'Emissions', info: 'Debug priority for UI preview', min: 0, max: 1, order: 1, start: 0.4, end: 0.2, value: 0.3, delta: 0.1, direction: '<=' },
//   { name: 'Reliability', info: 'Debug priority for UI preview', min: 50, max: 100, order: 2, start: 70, end: 82, value: 76, delta: 6, direction: '>=' },
//   { name: 'Comfort', info: 'Debug priority for UI preview', min: 0, max: 10, order: 3, start: 4, end: 7, value: 5.5, delta: 1, direction: '==' },
//   { name: 'Travel Time', info: 'Debug priority for UI preview', min: 5, max: 60, order: 4, start: 35, end: 22, value: 28, delta: 4, direction: '<=' },
//   { name: 'Safety', info: 'Debug priority for UI preview', min: 0, max: 100, order: 5, start: 65, end: 88, value: 76, delta: 8, direction: '>=' },
//   { name: 'Flexibility', info: 'Debug priority for UI preview', min: 0, max: 1, order: 6, start: 0.3, end: 0.75, value: 0.52, delta: 0.12, direction: '>=' },
// ])

/** @type {import('vue').Ref<Priority[] | null>} */
const prioritiesSnapshot = ref(null)

/** @type {import('vue').Ref<Array<{code: string, message: string}>>} */
const messages = ref([])

/** @type {import('vue').Ref<string>} */
const emit = defineEmits(['navigated'])

/** @type {import('vue').Ref<string>} */
const objLabel = ref('')

const loadInitData = async () => {
  const data = await get('/init')
  objLabel.value = String(data?.obj_label ?? '')

  const dimensions = data?.dimensions
  const point = data?.point

  if (!dimensions || typeof dimensions !== 'object' || Array.isArray(dimensions)) {
    throw new Error('Invalid init response: dimensions must be an object')
  }

  if (!point || typeof point !== 'object' || Array.isArray(point)) {
    throw new Error('Invalid init response: point must be an object')
  }

  priorities.value = Object.entries(dimensions).map(([name, range], index) => {
    const min = Number(range?.min)
    const max = Number(range?.max)

    if (Number.isNaN(min) || Number.isNaN(max)) {
      throw new Error(`Invalid init response: range for ${name} must contain numeric min/max`)
    }

    const start = Number(point[name])

    if (Number.isNaN(start)) {
      throw new Error(`Invalid init response: point for ${name} must be numeric`)
    }

    return {
      name,
      info: '',
      min,
      max,
      order: index,
      start,
      end: start,
      value: start,
      delta: 0,
      direction: '',
    }
  })

  const objIdx = priorities.value.findIndex((p) => p.name === objLabel.value)
  if (objIdx > 0) {
    const [obj] = priorities.value.splice(objIdx, 1)
    priorities.value.unshift(obj)
    priorities.value.forEach((p, i) => { p.order = i })
  }

  for (const priority of priorities.value) {
    priority.direction = priority.name === objLabel.value ? '>=' : ''
  }
}

/**
 * @param {Priority[]} source
 * @returns {Priority[]}
 */
const clonePriorities = (source) => source.map((priority) => ({ ...priority }))

const navigate = async () => {
  const constraints = {}

  for (const priority of priorities.value) {
    constraints[priority.name] = {
      delta: Number(priority.delta),
      direction: priority.direction,
      value: Number(priority.value),
    }
  }

  const data = await post('/navigate', { constraints })
  const nextConstraints = data.constraints
  const nextPoint = data.point
  const nextChanges = data.changes

  for (const priority of priorities.value) {
    const currentValue = Number(priority.value)
    const row = nextConstraints[priority.name]

    priority.delta = Number(row.delta)
    priority.direction = row.direction
    priority.start = currentValue
    priority.end = Number(nextPoint[priority.name])
  }

  messages.value = nextChanges
  path.value = data.breakpoints

  const firstPoint = data.breakpoints[0]?.point
  if (firstPoint) {
    const objPriority = priorities.value.find((p) => p.name === objLabel.value)
    if (objPriority) {
      objPriority.start = Number(firstPoint[objLabel.value])
    }
  }

  prioritiesSnapshot.value = clonePriorities(priorities.value)
  emit('navigated')
}

const reset = () => {
  if (!prioritiesSnapshot.value) return
  priorities.value = clonePriorities(prioritiesSnapshot.value)
}


const clampDeltas = () => {
  for (const priority of priorities.value) {
    if (!priority.direction) continue
    const ref = props.active ? priority.value : priority.start
    const { min, max } = priority
    let maxDelta
    if (priority.direction === '>=') maxDelta = max - ref
    else if (priority.direction === '<=') maxDelta = ref - min
    else if (priority.direction === '==') maxDelta = Math.min(max - ref, ref - min)
    if (maxDelta !== undefined && priority.delta > maxDelta) {
      priority.delta = Math.max(0, maxDelta)
    }
  }
}

watchEffect(() => {
  const breakpoints = path.value
  for (const priority of priorities.value) {
    if (priority.name === objLabel.value && breakpoints && breakpoints.length > 1) {
      const beta = props.beta
      let lo = breakpoints[0]
      let hi = breakpoints[breakpoints.length - 1]
      for (let i = 0; i < breakpoints.length - 1; i++) {
        if (beta >= breakpoints[i].beta && beta <= breakpoints[i + 1].beta) {
          lo = breakpoints[i]
          hi = breakpoints[i + 1]
          break
        }
      }
      const t = hi.beta === lo.beta ? 0 : (beta - lo.beta) / (hi.beta - lo.beta)
      priority.value = Number(lo.point[priority.name]) + (Number(hi.point[priority.name]) - Number(lo.point[priority.name])) * t
    } else {
      priority.value = priority.start + (priority.end - priority.start) * props.beta
    }
  }
})

watch(
  () => priorities.value.map((p) => ({ delta: p.delta, direction: p.direction })),
  clampDeltas,
  { deep: true },
)

watch(() => props.active, (active) => {
  if (!active) reset()
  else clampDeltas()
})

const draggedIndex = ref(null)
const isDragReady = ref(false)

const onDragStart = (index, event) => {
  if (!isDragReady.value) { event.preventDefault(); return }
  draggedIndex.value = index
  event.dataTransfer.effectAllowed = 'move'
}

const onDragOver = (event) => { event.preventDefault() }

const onDrop = (index) => {
  if (draggedIndex.value === null || draggedIndex.value === index) return
  const list = visiblePriorities.value
  const item = list.splice(draggedIndex.value, 1)[0]
  list.splice(index, 0, item)
  list.forEach((p, i) => { p.order = i })
  if (list[0]?.direction === '==') list[0].direction = ''
  draggedIndex.value = null
  isDragReady.value = false
}

const onDragEnd = () => {
  draggedIndex.value = null
  isDragReady.value = false
}

const visiblePriorities = computed(() => priorities.value)
const normalize = ref(false)

onMounted(() => {
  loadInitData()
    .then(() => navigate().catch((error) => {
      console.error('Failed initial navigate:', error)
    }))
    .catch((error) => {
      console.error('Failed to load init data:', error)
      objLabel.value = ''
      priorities.value = []
    })
})
</script>

<template>
  <section class="pane" :class="{ 'is-grayed': !active }">
    <div class="window-box main-box">
      <div class="box-header">
        <div class="header-title-wrap">
          <span class="header-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M12,2A10,10 0 0,0 2,12A10,10 0 0,0 12,22A10,10 0 0,0 22,12A10,10 0 0,0 12,2M12,4A8,8 0 0,1 20,12A8,8 0 0,1 12,20A8,8 0 0,1 4,12A8,8 0 0,1 12,4M17,7L13.5,13.5L7,17L10.5,10.5L17,7M12,11A1,1 0 0,0 11,12A1,1 0 0,0 12,13A1,1 0 0,0 13,12A1,1 0 0,0 12,11Z" />
            </svg>
          </span>
          <h3 class="header-title">Navigation</h3>
        </div>
        <div class="header-actions" :style="{ visibility: active ? 'visible' : 'hidden' }">
          <button
            type="button"
            class="normalize-toggle"
            :class="{ 'is-normalized': normalize }"
            @click="normalize = !normalize"
            aria-label="Toggle normalized display"
          >
            <span class="normalize-track">
              <span class="abs-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                  <line x1="9" y1="4" x2="9" y2="20" />
                  <line x1="15" y1="4" x2="15" y2="20" />
                  <line x1="4" y1="9" x2="20" y2="9" />
                  <line x1="4" y1="15" x2="20" y2="15" />
                </svg>
              </span>
              <span class="pct-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="7" cy="7" r="2.5" />
                  <circle cx="17" cy="17" r="2.5" />
                  <line x1="19" y1="5" x2="5" y2="19" />
                </svg>
              </span>
            </span>
            <span class="normalize-thumb" aria-hidden="true">
              <svg v-if="!normalize" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <line x1="9" y1="4" x2="9" y2="20" />
                <line x1="15" y1="4" x2="15" y2="20" />
                <line x1="4" y1="9" x2="20" y2="9" />
                <line x1="4" y1="15" x2="20" y2="15" />
              </svg>
              <svg v-else viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="7" cy="7" r="2.5" />
                <circle cx="17" cy="17" r="2.5" />
                <line x1="19" y1="5" x2="5" y2="19" />
              </svg>
            </span>
          </button>
          <button class="action-btn primary" type="button" @click="navigate">Explore</button>
          <button class="action-btn secondary" type="button" @click="reset">Reset</button>
        </div>
      </div>

      <div class="priority-content">
        <PriorityItem
          v-for="(priority, index) in visiblePriorities"
          :key="`${priority.name}-${index}`"
          class="priority-list-item"
          :draggable="active && isDragReady"
          :class="{ 'is-dragging': draggedIndex === index }"
          :name="priority.name"
          :order="priority.order"
          :min="priority.min"
          :max="priority.max"
          :start="priority.start"
          :end="priority.end"
          :info="priority.info"
          :value="priority.value"
          v-model:delta="priority.delta"
          v-model:direction="priority.direction"
          :disabled="!active"
          :normalize="normalize"
          :objective="priority.name === objLabel"
          @handle-mousedown="active && (isDragReady = true)"
          @dragstart="onDragStart(index, $event)"
          @dragover="onDragOver"
          @drop="onDrop(index)"
          @dragend="onDragEnd"
        />
      </div>
    </div>

    <div class="window-box message-box-wrap">
      <MessageBox :messages="messages" />
    </div>
  </section>
</template>

<style scoped>
.pane {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
  overflow: hidden;
  gap: 8px;
}

.window-box {
  background: #ffffff;
  border-radius: 4px;
  border: 1px solid #e3e3e3;
}

.main-box {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
  padding: 16px;
}

.box-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
}

.header-title-wrap {
  display: flex;
  align-items: center;
  gap: 6px;
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

.header-actions {
  display: flex;
  gap: 8px;
}

.action-btn {
  border: 1px solid transparent;
  border-radius: 4px;
  padding: 4px 10px;
  font-size: 0.875rem;
  cursor: pointer;
}

.action-btn.primary {
  background: #3e8ed0;
  color: #ffffff;
}

.action-btn.secondary {
  background: #ffffff;
  color: #3e8ed0;
  border-color: #3e8ed0;
}

.normalize-toggle {
  width: 56px;
  height: 26px;
  padding: 2px;
  border: none;
  border-radius: 13px;
  background-color: #f0f0f0;
  position: relative;
  cursor: pointer;
  box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
}

.normalize-track {
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 100%;
  padding: 0 6px;
  color: #888888;
}

.abs-icon,
.pct-icon {
  display: inline-flex;
  transition: opacity 0.3s;
  opacity: 0.3;
}

.normalize-toggle:not(.is-normalized) .abs-icon {
  opacity: 1;
}

.normalize-toggle.is-normalized .pct-icon {
  opacity: 1;
}

.normalize-thumb {
  position: absolute;
  top: 2px;
  left: 2px;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  background-color: #ffffff;
  color: #3e8ed0;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
  transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.normalize-toggle.is-normalized .normalize-thumb {
  transform: translateX(30px);
}

.priority-content {
  flex-grow: 1;
  overflow-y: auto;
  overflow-x: hidden;
  padding: 10px 10px 0 10px;
  margin: -10px -10px 0 -10px;
}

.priority-content {
  min-height: 0;
}

.priority-list-item {
  margin-bottom: 0.5rem;
  transition: opacity 0.2s ease;
}

.priority-list-item.is-dragging {
  opacity: 0.4;
}

.message-box-wrap {
  padding: 16px;
}

.is-grayed .window-box {
  background: #eeeeee;
}
</style>
