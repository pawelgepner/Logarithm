/* SPDX-License-Identifier: MIT
 * Platform functions of the logq benchmark for the Arduino build
 * (same as in ../../main/main.c). Author: Pawel Poczekajlo. */
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"

uint32_t bench_cycles(void) { return (uint32_t)esp_cpu_get_cycle_count(); }
void bench_irq_off(void)    { portDISABLE_INTERRUPTS(); }
void bench_irq_on(void)     { portENABLE_INTERRUPTS(); }
void bench_pause(void)      { vTaskDelay(pdMS_TO_TICKS(20)); }   /* keeps the watchdog fed */
