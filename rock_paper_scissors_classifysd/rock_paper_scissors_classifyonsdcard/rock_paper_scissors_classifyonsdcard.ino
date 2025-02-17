#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
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

namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* model_input = nullptr;
  int input_length;

  constexpr int kTensorArenaSize = 71*1028 + 128;
  uint8_t tensor_arena[kTensorArenaSize];

  // Deklariere das Array tofData hier
  float tofData[64];
}

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
File dataFile;

SPIClass sdspi = SPIClass();  // SPI für SD-Karte

void setup() {
  Serial.begin(115200);

  // Initialisiere SD-Karte mit SPIClass
  sdspi.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI, SD_CS_PIN);  // SPI-Pins anpassen, falls nötig
  if (!SD.begin(SD_CS_PIN, sdspi)) {
    Serial.println("Fehler beim Initialisieren der SD-Karte!");
    return;
  }

  Serial.println("SD-Karte erfolgreich initialisiert!");

  // Initialisiere TensorFlow Lite
  model = tflite::GetModel(get_model_data());
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Ungültige Modelldatei!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;
  interpreter->AllocateTensors();

  // Erhalte Eingabetensor
  model_input = interpreter->input(0);

  // Initialisiere Display
  display.begin(SSD1306_SWITCHCAPVCC, 0x3D);
  display.display();
  Serial.println("Display initialisiert!");

  delay(100);
  display.clearDisplay();
}

void readDataFromSD(File &dataFile) {
  int i = 0;
  while (dataFile.available() && i < 64) {
    tofData[i] = dataFile.parseFloat();  // Ein Wert von der Datei in den Array einlesen
    i++;
  }
}

void classifyAndDisplay() {
  for (int i = 0; i < 64; i++) {
    model_input->data.f[i] = tofData[i];  // Die Daten in den Eingabetensor laden
  }

  if (interpreter->Invoke() == kTfLiteOk) {
    const float* output = interpreter->output(0)->data.f;

    // Klassifikationswerte anzeigen
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(WHITE, BLACK);
    display.setCursor(10, 10);
    display.printf("Rock: %.2f\n", output[1]);
    display.setCursor(10, 30);
    display.printf("Paper: %.2f\n", output[0]);
    display.setCursor(10, 50);
    display.printf("Scissors: %.2f\n", output[2]);

    display.display();
  } else {
    display.setCursor(10, 30);
    display.println("Error!");
    display.display();
  }
}

void loop() {
  // Öffne die Datei von der SD-Karte
  dataFile = SD.open("tof_capture.csv");
  if (dataFile) {
    Serial.println("Datei erfolgreich geöffnet!");

    // Lese die Daten von der SD-Karte
    readDataFromSD(dataFile);
    dataFile.close();

    // Führe die Klassifikation aus und zeige das Ergebnis an
    classifyAndDisplay();
  } else {
    Serial.println("Fehler beim Öffnen der Datei!");
  }

  delay(5000);  // Warte 5 Sekunden, bevor der Vorgang wiederholt wird
}
