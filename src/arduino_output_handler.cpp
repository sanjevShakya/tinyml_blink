#include "output_handler.h"

#include "Arduino.h"
#include "constants.h"

int led = LED_BUILTIN;

bool initialized = false;

void HandleOutput(tflite::ErrorReporter *error_reporter, float x_value, float y_value)
{
    if (!initialized)
    {
        pinMode(led, OUTPUT);
        initialized = true;
    }

    int brightness = (int)(127.5f * (y_value + 1));
    analogWrite(led, brightness);
    Serial.println(brightness);
    // TF_LITE_REPORT_ERROR(error_reporter, "%d\n", brightness);
}