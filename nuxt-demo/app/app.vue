<template>
  <div>
    <NuxtRouteAnnouncer />
    <NuxtWelcome />
  </div>
</template>
<script setup lang="ts">
import DemoWorker from './demo.worker.ts?worker';

const demoWorker = new DemoWorker();
</script>
