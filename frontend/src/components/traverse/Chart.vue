<script setup>
import { onMounted, onUnmounted, ref, watch } from 'vue'
import * as echarts from 'echarts'
import { get, post } from '@/shared/api'

const props = defineProps({
  active: {
    type: Boolean,
    required: true,
  },
  beta: {
    type: Number,
    required: true,
  },
  path: {
    type: Array,
    required: true,
  },
})

/** @type {import('vue').Ref<string>} */
const objLabel = ref('')

/** @type {import('vue').Ref<string[]>} */
const types = ref([])
/** @type {import('vue').Ref<string>} */
const type = ref('')

const plotRef = ref(null)
let chartInstance = null

/**
 * @typedef {{ type: string, betas: number[], x_dim: number[], data: Array }} PlotData
 * @type {import('vue').Ref<PlotData | null>}
 */
const plotData = ref(null)

const interpolate = (beta, pd) => {
  const betas = pd.betas
  let lo = 0
  let hi = betas.length - 1

  for (let i = 0; i < betas.length - 1; i++) {
    if (beta >= betas[i] && beta <= betas[i + 1]) {
      lo = i
      hi = i + 1
      break
    }
  }

  const t = betas[hi] === betas[lo] ? 0 : (beta - betas[lo]) / (betas[hi] - betas[lo])
  const loData = pd.data[lo]
  const hiData = pd.data[hi]

  if (Array.isArray(loData)) {
    return loData.map((v, i) => v + (hiData[i] - v) * t)
  }

  const result = {}
  for (const key of Object.keys(loData)) {
    result[key] = loData[key].map((v, i) => v + (hiData[key][i] - v) * t)
  }
  return result
}

const computeYMax = (pd) => {
  let max = -Infinity
  for (const slice of pd.data) {
    if (Array.isArray(slice)) {
      for (const v of slice) if (v > max) max = v
    } else {
      const keys = Object.keys(slice)
      const len = slice[keys[0]]?.length ?? 0
      if (pd.type === 'stacked_bar') {
        for (let i = 0; i < len; i++) {
          const sum = keys.reduce((s, k) => s + (slice[k][i] ?? 0), 0)
          if (sum > max) max = sum
        }
      } else {
        for (const k of keys) for (const v of slice[k]) if (v > max) max = v
      }
    }
  }
  return max === -Infinity ? undefined : max
}

const renderChart = () => {
  if (!plotRef.value) return
  if (!chartInstance) chartInstance = echarts.init(plotRef.value)

  if (!plotData.value) {
    chartInstance.setOption({ series: [] }, true)
    return
  }

  const pd = plotData.value
  const yMax = computeYMax(pd)
  const interpolated = interpolate(props.beta, pd)

  const isTime = pd.type === 'timeseries' || pd.type === 'stacked_timeseries'
  const isBar = pd.type === 'bar' || pd.type === 'stacked_bar'
  const isStackedBar = pd.type === 'stacked_bar'

  // x_dim timestamps are in nanoseconds — convert to milliseconds for JS/echarts
  const xData = isTime ? pd.x_dim.map((ns) => ns / 1e6) : pd.x_dim

  let series
  if (Array.isArray(interpolated)) {
    series = [{
      name: type.value,
      type: isBar ? 'bar' : 'line',
      data: isTime ? xData.map((x, i) => [x, interpolated[i]]) : interpolated,
      lineStyle: { width: 1.5 },
      symbol: 'none',
    }]
  } else {
    series = Object.entries(interpolated).map(([name, values]) => ({
      name,
      type: isBar ? 'bar' : 'line',
      stack: isStackedBar ? 'total' : undefined,
      data: isTime ? xData.map((x, i) => [x, values[i]]) : values,
      lineStyle: { width: 1 },
      symbol: 'none',
    }))
  }

  chartInstance.setOption(
    {
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: isBar ? 'shadow' : 'line' },
        valueFormatter: fmtNum,
      },
      legend: { type: 'scroll', bottom: 0, textStyle: { fontSize: 11 } },
      dataZoom: isTime
        ? [
            { type: 'inside', xAxisIndex: 0 },
            { type: 'slider', xAxisIndex: 0, bottom: 24, height: 16, borderColor: '#d1d5db', fillerColor: 'rgba(72,95,199,0.1)', handleStyle: { color: '#485fc7' } },
          ]
        : [],
      grid: { top: 8, right: 16, bottom: isTime ? 72 : isBar ? 64 : 48, left: 64 },
      xAxis: {
        type: isTime ? 'time' : 'category',
        data: isTime ? undefined : pd.x_dim,
        axisLabel: {
          color: '#6b7280',
          fontSize: 11,
          interval: isTime ? 'auto' : 0,
          rotate: isBar ? 30 : 0,
        },
        axisLine: { lineStyle: { color: '#d1d5db' } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        axisLabel: { color: '#6b7280', fontSize: 11, formatter: yAxisFormatter },
        max: yMax,
        axisLine: { lineStyle: { color: '#d1d5db' } },
        splitLine: { lineStyle: { color: '#f3f4f6' } },
      },
      series,
    },
    true,
  )
}

const fmtNum = (v) => {
  const num = Number(v)
  if (!Number.isFinite(num)) return String(v)
  const abs = Math.abs(num)
  if (abs === 0) return '0'
  if (abs >= 1e9) return `${parseFloat((num / 1e9).toFixed(2))}G`
  if (abs >= 1e6) return `${parseFloat((num / 1e6).toFixed(2))}M`
  if (abs >= 1e3) return `${parseFloat((num / 1e3).toFixed(2))}k`
  return parseFloat(num.toPrecision(4)).toString()
}

const yAxisFormatter = fmtNum

const getObjValueAtBeta = (beta) => {
  if (!props.path || props.path.length === 0) return null
  const data = props.path.map((bp) => [bp.beta, Number(bp.point[objLabel.value])])
  if (beta <= data[0][0]) return data[0][1]
  if (beta >= data[data.length - 1][0]) return data[data.length - 1][1]
  for (let i = 0; i < data.length - 1; i++) {
    if (beta >= data[i][0] && beta <= data[i + 1][0]) {
      const t = (beta - data[i][0]) / (data[i + 1][0] - data[i][0])
      return data[i][1] + t * (data[i + 1][1] - data[i][1])
    }
  }
  return null
}

const renderObjLabelChart = () => {
  if (!chartInstance || !props.path || props.path.length < 2) return

  const seriesData = props.path.map((bp) => [bp.beta, Number(bp.point[objLabel.value])])
  const values = seriesData.map((d) => d[1])
  const yMax = Math.max(...values)
  const yMin = Math.min(...values)
  const pad = (yMax - yMin) * 0.1 || Math.abs(yMax) * 0.1 || 1

  chartInstance.setOption(
    {
      tooltip: { trigger: 'axis', valueFormatter: fmtNum },
      legend: { show: false },
      dataZoom: [],
      grid: { top: 8, right: 16, bottom: 48, left: 64 },
      xAxis: {
        type: 'value',
        min: 0,
        max: 1,
        name: 'β',
        nameLocation: 'end',
        axisLabel: { color: '#6b7280', fontSize: 11 },
        axisLine: { lineStyle: { color: '#d1d5db' } },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        min: yMin - pad,
        max: yMax + pad,
        axisLabel: { color: '#6b7280', fontSize: 11, formatter: yAxisFormatter },
        axisLine: { lineStyle: { color: '#d1d5db' } },
        splitLine: { lineStyle: { color: '#f3f4f6' } },
      },
      series: [
        {
          name: objLabel.value,
          type: 'line',
          data: seriesData,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: { color: '#485fc7', width: 2 },
          itemStyle: { color: '#485fc7' },
        },
        {
          name: objLabel.value,
          type: 'scatter',
          data: (() => { const y = getObjValueAtBeta(props.beta); return y !== null ? [[props.beta, y]] : [] })(),
          symbolSize: 10,
          animation: false,
          itemStyle: { color: '#48c78e' },
          z: 10,
        },
      ],
    },
    true,
  )
}

const fetchPlotData = async () => {
  if (!type.value || !props.path || props.path.length === 0) return
  if (type.value === objLabel.value) {
    renderObjLabelChart()
    return
  }
  const breakpoints = props.path.map((bp) => ({ beta: bp.beta, alpha: bp.alpha }))
  const data = await post('/plot_data', { name: type.value, breakpoints })
  plotData.value = data
  renderChart()
}

const resizeChart = () => {
  if (chartInstance) chartInstance.resize()
}


watch(() => type.value, () => {
  if (type.value === objLabel.value) renderObjLabelChart()
  else fetchPlotData()
})

watch(() => props.path, () => {
  if (type.value === objLabel.value) renderObjLabelChart()
  else fetchPlotData()
}, { deep: false })

watch(() => props.beta, () => {
  if (type.value !== objLabel.value) {
    renderChart()
  } else if (chartInstance) {
    const y = getObjValueAtBeta(props.beta)
    chartInstance.setOption({ series: [{}, { animation: false, data: y !== null ? [[props.beta, y]] : [] }] })
  }
})

onMounted(async () => {
  const initData = await get('/init_plot')
  objLabel.value = String(initData?.obj_label ?? '')
  const datasets = Array.isArray(initData?.datasets) ? initData.datasets : []
  datasets.forEach((t) => { if (!types.value.includes(t)) types.value.push(t) })
  if (objLabel.value && !types.value.includes(objLabel.value)) types.value.unshift(objLabel.value)
  if (types.value.length > 0) type.value = types.value[0]
  if (!chartInstance && plotRef.value) chartInstance = echarts.init(plotRef.value)
  window.addEventListener('resize', resizeChart)
})

onUnmounted(() => {
  window.removeEventListener('resize', resizeChart)
  if (chartInstance) {
    chartInstance.dispose()
    chartInstance = null
  }
})
</script>

<template>
  <section class="chart-area">
    <div class="chart-header">
      <span class="header-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="22" height="22">
          <rect x="3" y="3" width="18" height="18" rx="1" fill="none" stroke="currentColor" stroke-width="2" />
          <rect x="6" y="15" width="3" height="4" fill="currentColor" />
          <rect x="10.5" y="11" width="3" height="8" fill="currentColor" />
          <rect x="15" y="8" width="3" height="11" fill="currentColor" />
        </svg>
      </span>
      <h3 class="header-title">Chart</h3>
      <select v-model="type" class="type-select" :disabled="!active || types.length === 0">
        <option
          v-for="t in types"
          :key="t"
          :value="t"
        >{{ t === objLabel ? `${t} ∗` : t }}</option>
      </select>
    </div>

    <div class="plot-wrap">
      <div ref="plotRef" class="plot-canvas"></div>
    </div>
  </section>
</template>

<style scoped>
.chart-area {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-height: 0;
}

.chart-header {
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
  flex: 1;
}

.type-select {
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: #ffffff;
  color: #1f2937;
  font-size: 0.9rem;
  padding: 0.35rem 0.6rem;
}

.plot-wrap {
  flex: 1;
  min-height: 0;
}

.plot-canvas {
  width: 100%;
  height: 100%;
  min-height: 220px;
}
</style>
