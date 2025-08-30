/*
  IMU Classifier → JSON over UART (deviceId/timestamp/prediction/probability)
  Board: Arduino Nano 33 BLE / BLE Sense
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
#include <math.h>

#include "model.h"

const char* DEVICE_ID = "NANO33BLE-001";     // <-- set your device id

const float accelerationThreshold = 2.5f;   // trigger threshold in g's
const int   numSamples = 119;

int samplesRead = numSamples;

// TFLM globals
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver     tflOpsResolver;

const tflite::Model*       tflModel        = nullptr;
tflite::MicroInterpreter*  tflInterpreter  = nullptr;
TfLiteTensor*              tflInputTensor  = nullptr;
TfLiteTensor*              tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Labels for output indices
const char* GESTURES[] = { "normal", "abnormal" };
#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(115200);
  while (!Serial) { /* wait for USB CDC */ }

  if (!IMU.begin()) {
    // Avoid non-JSON prints if a receiver is listening; only print on fatal
    Serial.println("{\"deviceId\":\"" + String(DEVICE_ID) + "\",\"timestamp\":0.0,"
                   "\"prediction\":\"error\",\"probability\":0.0}");
    while (1) {}
  }

  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("{\"deviceId\":\"" + String(DEVICE_ID) + "\",\"timestamp\":0.0,"
                   "\"prediction\":\"schema_mismatch\",\"probability\":0.0}");
    while (1) {}
  }

  tflInterpreter = new tflite::MicroInterpreter(
      tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  tflInterpreter->AllocateTensors();

  tflInputTensor  = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Wait for significant motion
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabsf(aX) + fabsf(aY) + fabsf(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0; // arm capture
        break;
      }
    }
  }

  // Capture exactly numSamples rows
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Normalize to [0,1], same as training
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0f) / 8.0f;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0f) / 8.0f;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0f) / 8.0f;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0f) / 4000.0f;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0f) / 4000.0f;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0f) / 4000.0f;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Run inference
        if (tflInterpreter->Invoke() != kTfLiteOk) {
          // Emit an error record in the same schema
          String json;
          json.reserve(96);
          json  = "{\"deviceId\":\"";
          json += DEVICE_ID;
          json += "\",\"timestamp\":";
          json += String((double)millis(), 3);
          json += ",\"prediction\":\"invoke_failed\",\"probability\":0.0}";
          Serial.println(json);
          break;
        }

        // Argmax over NUM_GESTURES
        int   bestIdx = 0;
        float bestProb = tflOutputTensor->data.f[0];
        for (int i = 1; i < NUM_GESTURES; i++) {
          float p = tflOutputTensor->data.f[i];
          if (p > bestProb) { bestProb = p; bestIdx = i; }
        }

        // Build the requested JSON: deviceId, timestamp (float-like), prediction, probability
        unsigned long ts = millis();
        String json;
        json.reserve(128);
        json  = "{\"deviceId\":\"";
        json += DEVICE_ID;
        json += "\",\"timestamp\":";
        json += String((double)ts, 3);               // e.g., 532178.000
        json += ",\"prediction\":\"";
        json += GESTURES[bestIdx];                   // "normal" or "abnormal"
        json += "\",\"probability\":";
        json += String(bestProb, 3);                 // 0.000 .. 1.000
        json += "}";

        Serial.println(json); // one JSON line per window
      }
    }
  }
}
