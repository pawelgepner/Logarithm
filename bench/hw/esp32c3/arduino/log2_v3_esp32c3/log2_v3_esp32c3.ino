/* SPDX-License-Identifier: MIT
 * Arduino IDE entry point of the logq benchmark for ESP32-C3.
 * Port of ../../main/main.c by Pawel Poczekajlo (Koszalin University of Technology).
 * Before building, copy the benchmark sources into this folder:
 *     python3 ../make_sketch.py          (see ../README.md)               */
#include <Arduino.h>
#include "sdkconfig.h"
#include "esp_idf_version.h"

extern "C" {
#include "bench_core.h"
}

#ifndef BOARD_NAME
#define BOARD_NAME "set BOARD_NAME to your board, e.g. Waveshare ESP32-C3-Zero"
#endif

void setup()
{
    Serial.begin(115200);
    delay(2000);                              /* time to open the serial monitor */

    Serial.print("ESP-IDF version: ");
    Serial.println(esp_get_idf_version());
    Serial.printf("Arduino-ESP32 core: %d.%d.%d\n", ESP_ARDUINO_VERSION_MAJOR,
                  ESP_ARDUINO_VERSION_MINOR, ESP_ARDUINO_VERSION_PATCH);
    Serial.print("Compiler: GCC ");
    Serial.println(__VERSION__);
#if defined(__OPTIMIZE_SIZE__)
    Serial.println("opt: -Os");
#elif defined(__OPTIMIZE__)
    Serial.println("opt: -O1/-O2/-O3");
#else
    Serial.println("opt: none (-O0/-Og)");
#endif
    Serial.print("Board: ");
    Serial.println(BOARD_NAME);

    int mhz = (int)getCpuFrequencyMhz();
    printf("logq benchmark, %s, %d MHz\n", CONFIG_IDF_TARGET, mhz);
    bench_run(CONFIG_IDF_TARGET, mhz);
}

void loop() { vTaskDelay(portMAX_DELAY); }
