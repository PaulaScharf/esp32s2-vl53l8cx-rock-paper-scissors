// Code generated by senseBox Blockly on Wed Dec 18 2024 16:43:26 GMT+0100 (Central European Standard Time)

#include <vl53l8cx.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h> // http://librarymanager/All#Adafruit_GFX_Library
#include <Adafruit_SSD1306.h> // http://librarymanager/All#Adafruit_SSD1306
#include <Adafruit_NeoPixel.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  int input_length;

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 71*1028 ;
  uint8_t tensor_arena[kTensorArenaSize];
}

Adafruit_NeoPixel rgb_led_1 = Adafruit_NeoPixel(1, 1, NEO_GRB + NEO_KHZ800);

VL53L8CX sensor_vl53l8cx(&Wire, -1, -1);


float tofData[64] = {0};

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define SENSOR_WIDTH 8
#define SENSOR_HEIGHT 8
#define DISPLAY_WIDTH 64  // Expanded display width
#define DISPLAY_HEIGHT 64
#define SCALE_FACTOR 8    // Scale 8x8 to 64x64
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

uint16_t sensorBitmap[SENSOR_WIDTH * SENSOR_HEIGHT];  // Original sensor data
uint8_t scaledBitmap[DISPLAY_WIDTH * DISPLAY_HEIGHT / 8];  // Scaled bitmap for SSD1306

// Grayscale patterns for 4 levels
uint8_t grayscalePatterns[6][8] = {
    {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},  // White
    {0xEE, 0xDD, 0xEE, 0xDD, 0xEE, 0xDD, 0xEE, 0xDD},  // Very Light Gray
    {0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55},  // Medium Gray
    {0xAA, 0x00, 0xAA, 0x00, 0xAA, 0x00, 0xAA, 0x00},  // Dark Gray
    {0x88, 0x00, 0x88, 0x00, 0x88, 0x00, 0x88, 0x00},  // Very Dark Gray
    {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}   // Black
};

bool getVl53l8cxBitmap() {
  VL53L8CX_ResultsData Result;
  uint8_t NewDataReady = 0;
  uint8_t status;

  status = sensor_vl53l8cx.check_data_ready(&NewDataReady);

  String dataStr = "";

  bool saw_something = false;

  if ((!status) && (NewDataReady != 0)) {
    sensor_vl53l8cx.get_ranging_data(&Result);

    for (int j = 0; j < SENSOR_HEIGHT; j++) {
      for (int k = 0; k < SENSOR_WIDTH; k++) {
        int index = j * SENSOR_WIDTH + k;

        if ((long)(&Result)->target_status[VL53L8CX_NB_TARGET_PER_ZONE * index] == 255)
        {
          tofData[j+k] = 0;
          dataStr += String(0) + ",";
          sensorBitmap[index] = 0;
        }
        else
        {
          long distance = (long)(&Result)->distance_mm[VL53L8CX_NB_TARGET_PER_ZONE * index];
          if (distance > 400) {
            distance = 400;  // Cap distance at 400 cm
          } else {
            saw_something = true;
          }
          sensorBitmap[index] = map(distance, 0, 400, 0, 5);  // Map distance to levels 0-5
          tofData[j+k] = distance / (float)400.0;
          dataStr += String(distance) + ",";
        }
      }
    }
  }
  return saw_something;
}

void scaleBitmapWithShades(uint16_t* inputBitmap, uint8_t* outputBitmap) {
  memset(outputBitmap, 0, DISPLAY_WIDTH * DISPLAY_HEIGHT / 8);  // Clear the output bitmap

  for (int y = 0; y < SENSOR_HEIGHT; y++) {
    for (int x = 0; x < SENSOR_WIDTH; x++) {
      int inputIndex = y * SENSOR_WIDTH + x;
      uint8_t* pattern = grayscalePatterns[inputBitmap[inputIndex]];  // Get pattern for shade

      // Scale each sensor pixel to an 8x8 block in the 64x64 bitmap
      for (int dy = 0; dy < SCALE_FACTOR; dy++) {
        for (int dx = 0; dx < SCALE_FACTOR; dx++) {
          int scaledX = x * SCALE_FACTOR + dx;
          int scaledY = y * SCALE_FACTOR + dy;
          int outputIndex = (scaledY * DISPLAY_WIDTH + scaledX) / 8;
          int bitPosition = 7 - (scaledX % 8);  // Bit position in the byte (MSB to LSB)

          // Get the corresponding bit from the pattern
          uint8_t patternRow = pattern[dy % 8];
          if (patternRow & (1 << (7 - (dx % 8)))) {
            outputBitmap[outputIndex] |= (1 << bitPosition);  // Set the bit
          }
        }
      }
    }
  }
}

void setup() {

  Serial.begin(115200);

  rgb_led_1.begin();
  rgb_led_1.setBrightness(100);

  // ToF
  Wire.begin();
  Wire.setClock(1000000); //Sensor has max I2C freq of 1MHz
  sensor_vl53l8cx.begin();
  sensor_vl53l8cx.init();
  sensor_vl53l8cx.set_ranging_frequency_hz(30);
  sensor_vl53l8cx.set_resolution(VL53L8CX_RESOLUTION_8X8);
  sensor_vl53l8cx.start_ranging();
  Wire.setClock(100000); // reduce clock speed again
  Serial.println("VL53L8CX Initialized!");

  // Tensorflow
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(get_model_data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.printf("Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This imports all operations, which is more intensive, than just importing the ones we need.
  // If we ever run out of storage with a model, we can check here to free some space
  static tflite::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();
  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);

  Serial.printf("model_input->dims->size: %i \n", model_input->dims->size);
  Serial.printf("model_input->dims->data[0]: %i \n", model_input->dims->data[0]);
  Serial.printf("model_input->dims->data[1]: %i \n", model_input->dims->data[1]);
  // Serial.printf("model_input->dims->data[2]: %i \n", model_input->dims->data[2]);
  Serial.printf("model_input->type: %i \n", model_input->type);
  if ((model_input->dims->size != 2) || (model_input->dims->data[1] != 64) ||
      // (model_input->dims->data[2] != 8) || 
      (model_input->type != kTfLiteFloat32)) {
    Serial.println(model_input->dims->size);
    Serial.println(model_input->dims->data[0]);
    Serial.println(model_input->dims->data[1]);
    Serial.println(model_input->type);
    Serial.println("Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);
  Serial.printf("input_length: %i \n", input_length);

  // display
  display.begin(SSD1306_SWITCHCAPVCC, 0x3D);
  display.display();
  Serial.println("Display Initialized!");
  delay(100);
  display.clearDisplay();
}

void drawClassification(float rockPercentage, float paperPercentage, float scissorsPercentage) {
  int barWidth = 10;  // Width of each bar
  int barSpacing = 3; // Space between bars
  int baseY = 51; // Bottom of the chart (SSD1306 uses (0,0) at top-left)
  int maxHeight = 50; // Maximum height of bars

  // Scale percentages to fit within maxHeight
  int rockHeight = int(rockPercentage * maxHeight);
  int paperHeight = int(paperPercentage * maxHeight);
  int scissorsHeight = int(scissorsPercentage * maxHeight);

  // X positions for each bar
  int xRock = 70;
  int xPaper = xRock + barWidth + barSpacing;
  int xScissors = xPaper + barWidth + barSpacing;

  // Draw bars
  display.fillRect(xRock, baseY - rockHeight, barWidth, rockHeight, SSD1306_WHITE);
  display.fillRect(xPaper, baseY - paperHeight, barWidth, paperHeight, SSD1306_WHITE);
  display.fillRect(xScissors, baseY - scissorsHeight, barWidth, scissorsHeight, SSD1306_WHITE);

  // Draw labels under bars
  display.setTextSize(1);
  display.setTextColor(WHITE, BLACK);
  display.setCursor(xRock + 2, baseY + 2);
  display.println("R");
  display.setCursor(xPaper + 2, baseY + 2);
  display.println("P");
  display.setCursor(xScissors + 2, baseY + 2);
  display.println("S");
}


void loop() {
  float rockPercentage = -1.0;
  float paperPercentage = -1.0;
  float scissorsPercentage = -1.0;
  bool saw_something = getVl53l8cxBitmap();
  scaleBitmapWithShades(sensorBitmap, scaledBitmap);
  for (int i = 0; i < 64; ++i) {
    model_input->data.f[i] = tofData[i];
  }
  // Run inference, and report any error.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status == kTfLiteOk)
  {
      const float *prediction_scores = interpreter->output(0)->data.f;
      rockPercentage = prediction_scores[0];
      paperPercentage = prediction_scores[1];
      scissorsPercentage = prediction_scores[2];
  }

  display.clearDisplay();
  drawClassification(rockPercentage, paperPercentage, scissorsPercentage);
  display.drawBitmap(0, 0, scaledBitmap, DISPLAY_WIDTH, DISPLAY_HEIGHT, SSD1306_WHITE);
  display.display();
}