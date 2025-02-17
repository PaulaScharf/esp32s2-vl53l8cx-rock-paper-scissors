#include <vl53l8cx.h>
#include <SPI.h>
#include <Wire.h>
#include <SD.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_NeoPixel.h>
#include <FS.h>
#include <TensorFlowLite.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define SD_CS_PIN 10
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define SENSOR_WIDTH 8
#define SENSOR_HEIGHT 8
#define DISPLAY_WIDTH 64
#define DISPLAY_HEIGHT 64
#define SCALE_FACTOR 8
#define OLED_RESET -1

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
VL53L8CX sensor_vl53l8cx(&Wire, -1, -1);
float tofData[64] = {0};
uint16_t sensorBitmap[SENSOR_WIDTH * SENSOR_HEIGHT];
uint8_t scaledBitmap[DISPLAY_WIDTH * DISPLAY_HEIGHT / 8];

String filename = "/tof_capture.csv"; // Dateiname für SD-Karte
File Data; // Dateiobjekt für SD-Karte

SPIClass sdspi = SPIClass(); // SPI für SD-Karte

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
constexpr int kTensorArenaSize = 71*1028;
uint8_t tensor_arena[kTensorArenaSize];
int input_length;

void saveToSD() {
    Data = SD.open(filename, FILE_WRITE);
    if (Data) {
        for (int i = 0; i < 64; i++) {
            Data.print(tofData[i]);
            if (i % 8 == 7) {
                Data.print("\n");
            } else {
                Data.print(", ");
            }
        }
        Data.close();
        Serial.println("Data saved to SD card");
    } else {
        Serial.println("Error opening file for writing");
    }
}

bool captureFrame() {
    VL53L8CX_ResultsData Result;
    uint8_t NewDataReady = 0;
    uint8_t status = sensor_vl53l8cx.check_data_ready(&NewDataReady);
    bool detected = false;
    
    if (!status && NewDataReady) {
        sensor_vl53l8cx.get_ranging_data(&Result);
        for (int j = 0; j < SENSOR_HEIGHT; j++) {
            for (int k = 0; k < SENSOR_WIDTH; k++) {
                int index = j * SENSOR_WIDTH + k;
                if (Result.target_status[VL53L8CX_NB_TARGET_PER_ZONE * index] == 255) {
                    tofData[index] = 0;
                    sensorBitmap[index] = 0;
                } else {
                    long distance = Result.distance_mm[VL53L8CX_NB_TARGET_PER_ZONE * index];
                    distance = min(distance, (long)400);
                    sensorBitmap[index] = map(distance, 0, 400, 0, 5);
                    tofData[index] = distance / 200.0;
                    detected = true;
                }
            }
        }
    }
    return detected;
}

void classifyAndDisplay() {
    display.clearDisplay();  // Clear the display to redraw everything
    display.setTextSize(1);  // Set text size to 1 (default)
    display.setTextColor(WHITE, BLACK);  // White text on black background
    
    // Set initial vertical position for text
    int yPosition = 10;
    
    // Copy the ToF sensor data into model input
    for (int i = 0; i < 64; i++) {
        model_input->data.f[i] = tofData[i];
    }
    
    // Run the model and display results if successful
    if (interpreter->Invoke() == kTfLiteOk) {
        const float *output = interpreter->output(0)->data.f;
        
        // Display rock percentage
        display.setCursor(10, yPosition);
        display.printf("Rock: %.2f", output[1]);
        yPosition += 10; // Move down for the next line

        // Display paper percentage
        display.setCursor(10, yPosition);
        display.printf("Paper: %.2f", output[0]);
        yPosition += 10; // Move down for the next line
        
        // Display scissors percentage
        display.setCursor(10, yPosition);
        display.printf("Scissors: %.2f", output[2]);
    } else {
        // If there's an error, display an error message
        display.setCursor(10, 30);
        display.println("Error!");
    }
    
    display.display();  // Update the display with the new content
}


void showCapturingStatus() {
    display.clearDisplay();
    display.setTextSize(2);
    display.setTextColor(WHITE, BLACK);
    display.setCursor(10, 20);
    display.println("Erfassen...");
    display.display();
}

void setup() {
    Serial.begin(115200);
    Wire.begin();
    SPI.begin();  // Initialisiere SPI für die SD-Karte
    sensor_vl53l8cx.begin();
    sensor_vl53l8cx.init();
    sensor_vl53l8cx.set_ranging_frequency_hz(30);
    sensor_vl53l8cx.set_resolution(VL53L8CX_RESOLUTION_8X8);
    sensor_vl53l8cx.start_ranging();
    
    if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
        Serial.println("Display error");
    }

    // SD-Karteninitialisierung mit SPIClass
    sdspi.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI, SD_CS_PIN);
    if (!SD.begin(SD_CS_PIN, sdspi)) {
        Serial.println("SD Card error");
    } else {
        Serial.println("SD Card initialized successfully");
    }

    model = tflite::GetModel(get_model_data());
    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    interpreter->AllocateTensors();
    model_input = interpreter->input(0);
}

void loop() {
    showCapturingStatus(); // Status anzeigen
    if (captureFrame()) {
        saveToSD(); // Daten auf SD-Karte speichern
        classifyAndDisplay(); // Ergebnisse anzeigen
        while (true); // Stop nach einer Erfassung
    }
}