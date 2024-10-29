/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "person_detect_model_data.h"
#include "detection_responder.h"
#include "uart_communication.h"
#include "main_functions.h"
#include "image_provider.h"
#include "model_settings.h"
#include "esp_main.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "sdkconfig.h"
#include "esp_log.h"

#include <esp_heap_caps.h>
#include <esp_system.h>
#include <esp_timer.h>
#include <inttypes.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define LED_PIN GPIO_NUM_4
#define TX_OUTPUT_SIZE 1024
#define RX_INPUT_SIZE 1024
#define UART_PORT_NUM UART_NUM_2
#define NUM_INFERENCES 5

int inference_count = 0;
int detected_gestures[NUM_INFERENCES]; // almacena los labels de las detecciones mas prevalentes
//int detected_scores[NUM_INFERENCES];        // almacena los scores correspondientes

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.

  #ifdef CONFIG_IDF_TARGET_ESP32S3
    constexpr int scratchBufSize = 40 * 1024;
  #else
    constexpr int scratchBufSize = 0;
  #endif
    // An area of memory to use for input, output, and intermediate arrays.
    //constexpr int kTensorArenaSize = 81 * 1024 + scratchBufSize;
    constexpr int kTensorArenaSize = (96 * 96 * sizeof(uint)) + 330000;
    static uint8_t *tensor_arena;//[kTensorArenaSize]; // Maybe we should move this to external
}  // namespace

void determine_prevalent_gesture() {
  int counts[7] = {0, 0, 0, 0, 0, 0, 0};
  int prevalent_gesture = 6;
  int max_count = 0;

  // se cuentan las ocurrencias de cada gesto
  for (int i = 0; i < NUM_INFERENCES; i++) {
    counts[detected_gestures[i]]++;
  }

  // se determina el gesto mas prevalente
  for (int i = 0; i < NUM_INFERENCES - 1; i++) {
    if (counts[i] > max_count) {
      max_count = counts[i];
      prevalent_gesture = i;
    }
  }

  MicroPrintf("detected_gestures: %d, %d, %d, %d, %d", detected_gestures[0], detected_gestures[1], detected_gestures[2], detected_gestures[3], detected_gestures[4]);
  MicroPrintf("Gesto prevalente: %d (%d, %d, %d, %d, %d, %d)", prevalent_gesture, counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);

  char strGesture[12];
  sprintf(strGesture, "%d", prevalent_gesture);
  uart_send_data(strGesture);
}

// The name of this function is important for Arduino compatibility.
void setup() { // SETUP

  ESP_LOGI("SETUP", "Starting setup");

  // if (esp_psram_get_size() == 0) {
  //   printf("PSRAM not found\n");
  //   return;
  // }

  esp_rom_gpio_pad_select_gpio(LED_PIN);
  gpio_set_direction(LED_PIN, GPIO_MODE_OUTPUT);
  gpio_set_level(LED_PIN, 0);

  printf("Total heap size: %d\n", heap_caps_get_total_size(MALLOC_CAP_8BIT));
  printf("Free heap size: %d\n", heap_caps_get_free_size(MALLOC_CAP_8BIT));
  // printf("Total PSRAM size: %d\n", esp_psram_get_size());
  // printf("Free PSRAM size: %d\n", heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_person_detect_model_data);
  if (model == nullptr) {
    printf("Error: No se pudo cargar el modelo.\n");
    return;
  }
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Allocate tensor arena in PSRAM
  if (tensor_arena == NULL) {
    //tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    tensor_arena = (uint8_t *) heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM);
  }
  if (tensor_arena == NULL) {
    printf("Couldn't allocate memory of %d bytes\n", kTensorArenaSize);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
    static tflite::MicroMutableOpResolver<6> micro_op_resolver;
    micro_op_resolver.AddConv2D();
    micro_op_resolver.AddMaxPool2D();
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddReshape();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddQuantize();

  // Build an interpreter to run the model with.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  input = interpreter->input(0);
  if (input == nullptr) {
    printf("Error: No se pudo obtener el tensor de entrada.\n");
    return;
  }


#ifndef CLI_ONLY_INFERENCE
  // Initialize Camera
  TfLiteStatus init_status = InitCamera();
  if (init_status != kTfLiteOk) {
    MicroPrintf("InitCamera failed\n");
    return;
  }
#endif
}

#ifndef CLI_ONLY_INFERENCE
// The name of this function is important for Arduino compatibility.
void loop() { // LOOP
  MicroPrintf("Inference count: %d", inference_count);

  // uint8_t *display_buf = (uint8_t *) malloc(RX_INPUT_SIZE);

  // bzero(display_buf, RX_INPUT_SIZE);

  // int len = uart_read_bytes(UART_PORT_NUM, display_buf, RX_INPUT_SIZE, 1000 / portTICK_PERIOD_MS);

  // if (len > 0) {
  //   gpio_set_level(LED_PIN, 1);
  //   vTaskDelay(pdMS_TO_TICKS(1000));
  // }
  // else {
  //   gpio_set_level(LED_PIN, 0);
  // }

  // Get image from provider.
  if (kTfLiteOk != GetImage(kNumCols, kNumRows, kNumChannels, input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = interpreter->output(0);
  
  // Process the inference results.
  int k1_score = output->data.uint8[k1Index];
  int k2_score = output->data.uint8[k2Index];
  int k3_score = output->data.uint8[k3Index];
  int k4_score = output->data.uint8[k4Index];
  int k5_score = output->data.uint8[k5Index];
  int k7_score = output->data.uint8[k7Index];
  int kBlank_score = output->data.uint8[kBlankIndex];

  float k1_score_f = (k1_score - output->params.zero_point) * output->params.scale;
  float k2_score_f = (k2_score - output->params.zero_point) * output->params.scale;
  float k3_score_f = (k3_score - output->params.zero_point) * output->params.scale;
  float k4_score_f = (k4_score - output->params.zero_point) * output->params.scale;
  float k5_score_f = (k5_score - output->params.zero_point) * output->params.scale;
  float k7_score_f = (k7_score - output->params.zero_point) * output->params.scale;
  float kBlank_score_f = (kBlank_score - output->params.zero_point) * output->params.scale;

  int max_index = RespondToDetection(k1_score_f, k2_score_f, k3_score_f, k4_score_f, k5_score_f, k7_score_f, kBlank_score_f);

  // Guardar la clasificación más alta en detected_gestures
  detected_gestures[inference_count] = max_index;
  //detected_scores[inference_count] = scores[max_index];

  inference_count++;

  // Si hemos llegado a 5 inferencias, determinar el gesto prevalente
  if (inference_count == NUM_INFERENCES) {
    
    for (int i = 0; i < NUM_INFERENCES; i++) {
      //MicroPrintf("Gesto: %s, Score: %d", detected_gestures[i], detected_scores[i]);
      MicroPrintf("Gesto [%d]: %d", i + 1, detected_gestures[i]);
      
    }
    determine_prevalent_gesture();
    
    inference_count = 0; // Resetear el contador para la próxima ronda de inferencias
  }

  // Aquí puedes agregar el código para encender el LED
  // gpio_set_level(LED_PIN, 1); // Encender el LED
  vTaskDelay(pdMS_TO_TICKS(10)); // Espera 1000 ms (1 segundo)
  // gpio_set_level(LED_PIN, 0); // Apagar el LED
}
#endif

#if defined(COLLECT_CPU_STATS)
  long long total_time = 0;
  long long start_time = 0;
  extern long long softmax_total_time;
  extern long long dc_total_time;
  extern long long conv_total_time;
  extern long long fc_total_time;
  extern long long pooling_total_time;
  extern long long add_total_time;
  extern long long mul_total_time;
#endif

void run_inference(void *ptr) {

  /* Convert from uint8 picture data to int8 */
  for (int i = 0; i < kNumCols * kNumRows; i++) {
    input->data.uint8[i] = ((uint8_t *) ptr)[i];
    // print the image
    //printf("%d ", input->data.uint8[i]);
  }

#if defined(COLLECT_CPU_STATS)
  long long start_time = esp_timer_get_time();
#endif
  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

#if defined(COLLECT_CPU_STATS)
  long long total_time = (esp_timer_get_time() - start_time);
  printf("Total time = %lld\n", total_time / 1000);
  printf("Softmax time = %lld\n", softmax_total_time / 1000);
  printf("FC time = %lld\n", fc_total_time / 1000);
  printf("DC time = %lld\n", dc_total_time / 1000);
  printf("conv time = %lld\n", conv_total_time / 1000);
  printf("Pooling time = %lld\n", pooling_total_time / 1000);
  printf("add time = %lld\n", add_total_time / 1000);
  printf("mul time = %lld\n", mul_total_time / 1000);

  /* Reset times */
  total_time = 0;
  //softmax_total_time = 0;
  dc_total_time = 0;
  conv_total_time = 0;
  fc_total_time = 0;
  pooling_total_time = 0;
  add_total_time = 0;
  mul_total_time = 0;
#endif

  TfLiteTensor* output = interpreter->output(0);

  // // Process the inference results.
  // int8_t person_score = output->data.uint8[kPersonIndex];
  // int8_t no_person_score = output->data.uint8[kNotAPersonIndex];

  // float person_score_f = (person_score - output->params.zero_point) * output->params.scale;
  // float no_person_score_f = (no_person_score - output->params.zero_point) * output->params.scale;
  // RespondToDetection(person_score_f, no_person_score_f);

  int k1_score = output->data.uint8[k1Index];
  int k2_score = output->data.uint8[k2Index];
  int k3_score = output->data.uint8[k3Index];
  int k4_score = output->data.uint8[k4Index];
  int k5_score = output->data.uint8[k5Index];
  int k7_score = output->data.uint8[k7Index];
  int kBlank_score = output->data.uint8[kBlankIndex];

  float k1_score_f = (k1_score - output->params.zero_point) * output->params.scale;
  float k2_score_f = (k2_score - output->params.zero_point) * output->params.scale;
  float k3_score_f = (k3_score - output->params.zero_point) * output->params.scale;
  float k4_score_f = (k4_score - output->params.zero_point) * output->params.scale;
  float k5_score_f = (k5_score - output->params.zero_point) * output->params.scale;
  float k7_score_f = (k7_score - output->params.zero_point) * output->params.scale;
  float kBlank_score_f = (kBlank_score - output->params.zero_point) * output->params.scale;

  RespondToDetection(k1_score_f, k2_score_f, k3_score_f, k4_score_f, k5_score_f, k7_score_f, kBlank_score_f);
  // //vTaskDelay(8000 / portTICK_PERIOD_MS); // to avoid watchdog trigger

}
