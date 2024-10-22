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

#include "main_functions.h"

#include "detection_responder.h"
#include "image_provider.h"
#include "model_settings.h"
#include "person_detect_model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <esp_heap_caps.h>
#include <esp_timer.h>
#include <esp_log.h>
#include "esp_main.h"
// #include "esp_psram.h"

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

// The name of this function is important for Arduino compatibility.
void setup() {

  // if (esp_psram_get_size() == 0) {
  //   printf("PSRAM not found\n");
  //   return;
  // }

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
void loop() {
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
  int k10_score = output->data.uint8[k10Index];
  int k2_score = output->data.uint8[k2Index];
  int k3_score = output->data.uint8[k3Index];
  int k4_score = output->data.uint8[k4Index];
  int k5_score = output->data.uint8[k5Index];
  int kBlank_score = output->data.uint8[kBlankIndex];

  float k1_score_f = (k1_score - output->params.zero_point) * output->params.scale;
  float k10_score_f = (k10_score - output->params.zero_point) * output->params.scale;
  float k2_score_f = (k2_score - output->params.zero_point) * output->params.scale;
  float k3_score_f = (k3_score - output->params.zero_point) * output->params.scale;
  float k4_score_f = (k4_score - output->params.zero_point) * output->params.scale;
  float k5_score_f = (k5_score - output->params.zero_point) * output->params.scale;
  float kBlank_score_f = (kBlank_score - output->params.zero_point) * output->params.scale;

  RespondToDetection(k1_score_f, k10_score_f, k2_score_f, k3_score_f, k4_score_f, k5_score_f, kBlank_score_f);
  vTaskDelay(pdMS_TO_TICKS(2000)); // to avoid watchdog trigger

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
  int k10_score = output->data.uint8[k10Index];
  int k2_score = output->data.uint8[k2Index];
  int k3_score = output->data.uint8[k3Index];
  int k4_score = output->data.uint8[k4Index];
  int k5_score = output->data.uint8[k5Index];
  int kBlank_score = output->data.uint8[kBlankIndex];

  float k1_score_f = (k1_score - output->params.zero_point) * output->params.scale;
  float k10_score_f = (k10_score - output->params.zero_point) * output->params.scale;
  float k2_score_f = (k2_score - output->params.zero_point) * output->params.scale;
  float k3_score_f = (k3_score - output->params.zero_point) * output->params.scale;
  float k4_score_f = (k4_score - output->params.zero_point) * output->params.scale;
  float k5_score_f = (k5_score - output->params.zero_point) * output->params.scale;
  float kBlank_score_f = (kBlank_score - output->params.zero_point) * output->params.scale;

  RespondToDetection(k1_score_f, k10_score_f, k2_score_f, k3_score_f, k4_score_f, k5_score_f, kBlank_score_f);
  // vTaskDelay(8000 / portTICK_PERIOD_MS); // to avoid watchdog trigger

}
