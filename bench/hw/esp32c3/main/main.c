/* SPDX-License-Identifier: MIT
 * ESP-IDF entry point of the logq benchmark (tested target: ESP32-C3). */
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_cpu.h"
#include "esp_rom_sys.h"
#include "bench_core.h"

uint32_t bench_cycles(void) { return (uint32_t)esp_cpu_get_cycle_count(); }
void bench_irq_off(void) { portDISABLE_INTERRUPTS(); }
void bench_irq_on(void) { portENABLE_INTERRUPTS(); }
void bench_pause(void) { vTaskDelay(pdMS_TO_TICKS(20)); }   /* keeps the watchdog fed */

void app_main(void)
{
    vTaskDelay(pdMS_TO_TICKS(2000));          /* time to open the serial monitor */
    printf("logq benchmark, %s, %d MHz\n", CONFIG_IDF_TARGET, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
    bench_run(CONFIG_IDF_TARGET, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ);
}
