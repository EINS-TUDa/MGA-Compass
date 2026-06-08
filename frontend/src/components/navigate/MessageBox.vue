<script setup>
const props = defineProps({
  messages: {
    type: Array,
    default: () => [],
  },
})

const CODE_CLASS = {
  delta_reduced: 'warning',
  delta_widened: 'warning',
  infeasible_lb: 'danger',
  infeasible_ub: 'danger',
}

const cardClass = (code) => `is-${CODE_CLASS[code] ?? 'info'}`
</script>

<template>
  <section class="message-box">
    <div class="message-header">
      <span class="header-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="20" height="20">
          <path fill="currentColor" d="M20,2H4A2,2 0 0,0 2,4V22L6,18H20A2,2 0 0,0 22,16V4A2,2 0 0,0 20,2M20,16H5.17L4,17.17V4H20V16Z" />
        </svg>
      </span>
      <h3 class="header-title">Messages</h3>
    </div>

    <div class="messages-container">
      <p v-if="props.messages.length === 0" class="empty-state">No messages</p>
      <article
        v-for="(message, index) in props.messages"
        :key="index"
        class="message-window"
        :class="cardClass(message.code)"
      >
        <div class="message-body">
          {{ message.message }}
        </div>
      </article>
    </div>
  </section>
</template>

<style scoped>
.message-box {
  display: flex;
  min-height: 180px;
  flex-direction: column;
}

.message-header {
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

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 10px 10px 0 10px;
  margin: -10px -10px 0 -10px;
}

.empty-state {
  margin: 0;
  color: #9ca3af;
  text-align: center;
}

.message-window {
  margin: 0 0 8px;
  border-radius: 4px;
  border: 1px solid transparent;
}

.message-body {
  padding: 8px 10px;
  color: #1f2937;
  overflow-wrap: break-word;
  word-break: break-word;
}

.message-window.is-info {
  border-color: #bde0fe;
  background: #eff6ff;
}

.message-window.is-success {
  border-color: #b7e4c7;
  background: #effaf3;
}

.message-window.is-warning {
  border-color: #ffe8a3;
  background: #fff8e1;
}

.message-window.is-danger {
  border-color: #ffb3c1;
  background: #fff1f2;
}
</style>
