#include <vl53l8cx.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define SENSOR_WIDTH 8
#define SENSOR_HEIGHT 8
#define OLED_RESET -1
#define DURATION 5000
#define RESULT_DISPLAY_TIME 3000

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  constexpr int kTensorArenaSize = 71 * 1028 + 128;
  uint8_t tensor_arena[kTensorArenaSize];
}

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
VL53L8CX sensor_vl53l8cx(&Wire, -1, -1);
float tofData[64] = {0};
String choices[] = {"Rock", "Paper", "Scissors"};

void setup() {
  Serial.begin(115200);
  Wire.begin();
  sensor_vl53l8cx.begin();
  sensor_vl53l8cx.init();
  sensor_vl53l8cx.start_ranging();
  display.begin(SSD1306_SWITCHCAPVCC, 0x3D);
  display.display();
  delay(100);
  display.clearDisplay();

  model = tflite::GetModel(get_model_data());
  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();
  model_input = interpreter->input(0);
}

void showMessage(String message, int duration) {
  display.clearDisplay();
  display.setTextSize(2);
  display.setTextColor(WHITE, BLACK);
  display.setCursor(10, 20);
  display.println(message);
  display.display();
  delay(duration);
}

bool getVl53l8cxData() {
  VL53L8CX_ResultsData Result;
  uint8_t NewDataReady = 0;
  if (!sensor_vl53l8cx.check_data_ready(&NewDataReady) && NewDataReady) {
    sensor_vl53l8cx.get_ranging_data(&Result);
    for (int j = 0; j < SENSOR_HEIGHT; j++) {
      for (int k = 0; k < SENSOR_WIDTH; k++) {
        int index = j * SENSOR_WIDTH + k;
        long distance = (long)(&Result)->distance_mm[VL53L8CX_NB_TARGET_PER_ZONE * index];
        tofData[j + k] = distance / 400.0;
      }
    }
    return true;
  }
  return false;
}

String classifyGesture() {
  unsigned long startTime = millis();
  float rockSum = 0, paperSum = 0, scissorsSum = 0;
  int count = 0;
  
  while (millis() - startTime < DURATION) {
    int remainingTime = (DURATION - (millis() - startTime)) / 1000;
    showMessage("Time: " + String(remainingTime) + "s", 1000);
    
    if (getVl53l8cxData()) {
      for (int i = 0; i < 64; ++i) {
        model_input->data.f[i] = tofData[i];
      }
      if (interpreter->Invoke() == kTfLiteOk) {
        const float *scores = interpreter->output(0)->data.f;
        paperSum += scores[0];
        rockSum += scores[1];
        scissorsSum += scores[2];
        count++;
      }
    }
  }
  
  float avgRock = rockSum / count;
  float avgPaper = paperSum / count;
  float avgScissors = scissorsSum / count;
  return (avgRock > avgPaper && avgRock > avgScissors) ? "Rock" : (avgPaper > avgRock && avgPaper > avgScissors) ? "Paper" : "Scissors";
}

String determineWinner(String senseBoxChoice, String userChoice) {
  if (senseBoxChoice == userChoice) return "Draw!";
  if ((senseBoxChoice == "Rock" && userChoice == "Scissors") ||
      (senseBoxChoice == "Scissors" && userChoice == "Paper") ||
      (senseBoxChoice == "Paper" && userChoice == "Rock")) {
    return "SenseBox Wins!";
  }
  return "You Win!";
}

void loop() {
  String senseBoxChoice = choices[random(3)];
  showMessage("SenseBox: " + senseBoxChoice, 2000);
  String userChoice = classifyGesture();
  showMessage("You: " + userChoice, RESULT_DISPLAY_TIME);
  showMessage(determineWinner(senseBoxChoice, userChoice), RESULT_DISPLAY_TIME);
}
