<script setup>
import Chart from './traverse/Chart.vue'
import Slider from './traverse/Slider.vue'

defineProps({
  active: {
    type: Boolean,
    required: true,
  },
  path: {
    type: Array,
    required: true,
  },
})

const beta = defineModel('beta', {
  type: Number,
  required: true,
})
</script>

<template>
  <section class="pane" :class="{ 'is-grayed': !active }">
    <div class="window-box chart-box">
      <Chart :active="active" :beta="beta" :path="path" />
    </div>

    <div class="window-box slider-box">
      <Slider :active="active" v-model:beta="beta" />
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

.chart-box {
  display: flex;
  flex: 1;
  min-height: 0;
  flex-direction: column;
  padding: 16px;
}


.slider-box {
  padding: 16px;
}

.is-grayed .window-box {
  background: #eeeeee;
}
</style>
