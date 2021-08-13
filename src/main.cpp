#include <TensorFlowLite.h>
#include "main_functions.h"
#include "Arduino.h"

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "constants.h"
#include "model.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace
{
  tflite::ErrorReporter *error_reporter = nullptr;
  const tflite::Model *model = nullptr;
  tflite::MicroInterpreter *interpreter = nullptr;
  TfLiteTensor *input = nullptr;
  TfLiteTensor *output = nullptr;
  int inference_count = 0;

  constexpr int kTensorArenaSize = 2000;
  uint8_t tensor_arena[kTensorArenaSize];
} // namespace

void setup()
{
  tflite::InitializeTarget();

  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Model provided is schema version %d not equal to supported version %d", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::AllOpsResolver resolver;

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "AlocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  inference_count = 0;
}

void loop()
{
  float position = static_cast<float>(inference_count) / static_cast<float>(kInferencesPerCycle);
  float x = position * kXrange;

  int8_t x_quantized = x / input->params.scale + input->params.zero_point;

  input->data.int8[0] = x_quantized;

  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk)
  {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x: %f\n", static_cast<double>(x));
    return;
  }

  int8_t y_quantized = output->data.int8[0];

  float y = (y_quantized - output->params.zero_point) * output->params.scale;

  HandleOutput(error_reporter, x, y);
  // Serial.println(y);
  inference_count += 1;

  if (inference_count >= kInferencesPerCycle)
    inference_count = 0;
}