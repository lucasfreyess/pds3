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

/*
 * SPDX-FileCopyrightText: 2019-2023 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detection_responder.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "model_settings.h"

#include "esp_main.h"

#if DISPLAY_SUPPORT
#include "image_provider.h"
#include "bsp/esp-bsp.h"

// Camera definition is always initialized to match the trained detection model: 96x96 pix
// That is too small for LCD displays, so we extrapolate the image to 192x192 pix
#define IMG_WD (96 * 2)
#define IMG_HT (96 * 2)

static lv_obj_t *camera_canvas = NULL;
static lv_obj_t *person_indicator = NULL;
static lv_obj_t *label = NULL;

static void create_gui(void)
{
  bsp_display_start();
  bsp_display_backlight_on(); // Set display brightness to 100%
  bsp_display_lock(0);
  camera_canvas = lv_canvas_create(lv_scr_act());
  assert(camera_canvas);
  lv_obj_align(camera_canvas, LV_ALIGN_TOP_MID, 0, 0);

  person_indicator = lv_led_create(lv_scr_act());
  assert(person_indicator);
  lv_obj_align(person_indicator, LV_ALIGN_BOTTOM_MID, -70, 0);
  lv_led_set_color(person_indicator, lv_palette_main(LV_PALETTE_GREEN));

  label = lv_label_create(lv_scr_act());
  assert(label);
  lv_label_set_text_static(label, "Person detected");
  lv_obj_align_to(label, person_indicator, LV_ALIGN_OUT_RIGHT_MID, 20, 0);
  bsp_display_unlock();
}
#endif // DISPLAY_SUPPORT

void RespondToDetection(float k1_score, float k10_score, float k2_score, float k3_score, float k4_score, float k5_score, float kBlank_score) {
  
  int k1_score_int = (k1_score) * 100 + 0.5;
  int k10_score_int = (k10_score) * 100 + 0.5;
  int k2_score_int = (k2_score) * 100 + 0.5;
  int k3_score_int = (k3_score) * 100 + 0.5;
  int k4_score_int = (k4_score) * 100 + 0.5;
  int k5_score_int = (k5_score) * 100 + 0.5;
  int kBlank_score_int = (kBlank_score) * 100 + 0.5;

  int max_score = k1_score_int;
  int max_index = 1;
  int scores[kCategoryCount] = {k1_score_int, k10_score_int, k2_score_int, k3_score_int, k4_score_int, k5_score_int, kBlank_score_int};
  for (int i = 1; i < kCategoryCount; i++) {
    if (scores[i] > max_score) {
      max_score = scores[i];
      max_index = i;
    }
  }
  
  MicroPrintf("MAX SCORE: %d%% (Class: %s)", max_score, kCategoryLabels[max_index]);
  MicroPrintf("SCORES:\n     1: %d%% (%f%%)\n    10: %d%% (%f%%)\n     2: %d%% (%f%%)\n     3: %d%% (%f%%)\n     4: %d%% (%f%%)\n     5: %d%% (%f%%)\n Blank: %d%% (%f%%)\n\n",
              k1_score_int, k1_score, k10_score_int, k10_score, k2_score_int, k2_score, k3_score_int, k3_score, k4_score_int, k4_score, k5_score_int, k5_score, kBlank_score_int, kBlank_score);

}
