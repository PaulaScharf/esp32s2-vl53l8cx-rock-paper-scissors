// Code generated by senseBox Blockly on Wed Dec 18 2024 16:43:26 GMT+0100 (Central European Standard Time)

#include <vl53l8cx.h>
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h> // http://librarymanager/All#Adafruit_GFX_Library
#include <Adafruit_SSD1306.h> // http://librarymanager/All#Adafruit_SSD1306
#include <SPI.h>
#include <SD.h>
#include "FS.h"
#include <Adafruit_NeoPixel.h>

unsigned long lastSaveTime = 0; // Variable to store the last time data was saved

Adafruit_NeoPixel rgb_led_1 = Adafruit_NeoPixel(1, 1, NEO_GRB + NEO_KHZ800);

VL53L8CX sensor_vl53l8cx(&Wire, -1, -1);

String filename;
File Data;
SPIClass sdspi = SPIClass();


String tofData;
String class_name;

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
          dataStr += String(distance) + ",";
        }
      }
    }
  }
  tofData = dataStr;
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

void saveDataToSD()
{
  Data = SD.open(filename, FILE_APPEND);
  if (Data)
  {
    Data.print(class_name);
    Data.print(",");
    Data.println(tofData);
    Data.close();
  }
}

void setup() {
  class_name = "scissors";

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

  // display
  display.begin(SSD1306_SWITCHCAPVCC, 0x3D);
  display.display();
  Serial.println("Display Initialized!");
  delay(100);
  display.clearDisplay();

  // SD
  pinMode(SD_ENABLE, OUTPUT);
  digitalWrite(SD_ENABLE, LOW);
  sdspi.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI, VSPI_SS);
  display.print("Setup SD: ");
  display.display();
  filename = "/"+class_name+".csv";
  display.println(filename);
  display.display();
  SD.begin(VSPI_SS, sdspi);
  Data = SD.open(filename, FILE_WRITE);
  if (!Data)
  {
    display.println("No SD card found :(");
    display.display();
    while (1)
    {
      delay(10);
    }
  }
  Data.print("Class");
  Data.print(",");
  Data.println("ToF");
  Data.close();
  display.clearDisplay();
  Serial.println("SD Initialized!");
}

void drawTimer(unsigned long time) {
  display.setCursor(80,32);
  display.setTextSize(1);
  display.setTextColor(WHITE,BLACK);
  display.println(3-(time-(time%1000))/1000);
}

void drawWaiting() {
  display.setCursor(80,32);
  display.setTextSize(1);
  display.setTextColor(WHITE,BLACK);
  display.println("...");
}

void drawClassName() {
  display.setCursor(80,0);
  display.setTextSize(1);
  display.setTextColor(WHITE,BLACK);
  display.println("class:");
  display.setCursor(80,8);
  display.println(class_name);
}

void loop() {
  bool saw_something = getVl53l8cxBitmap();
  scaleBitmapWithShades(sensorBitmap, scaledBitmap);

  display.clearDisplay();
  if(!saw_something) {
    lastSaveTime = millis()-1000; // Update the last save time
    drawWaiting();
  } else {
    drawTimer(millis() - lastSaveTime);
  }
  drawClassName();
  display.drawBitmap(0, 0, scaledBitmap, DISPLAY_WIDTH, DISPLAY_HEIGHT, SSD1306_WHITE);
  display.display();


  if (millis() - lastSaveTime >= 3000)
  {
    rgb_led_1.setPixelColor(0, rgb_led_1.Color(255, 255, 255));
    rgb_led_1.show();
    saveDataToSD();          // Save data to SD card
    lastSaveTime = millis(); // Update the last save time
    rgb_led_1.setPixelColor(0, rgb_led_1.Color(0, 0, 0));
    rgb_led_1.show();
    delay(500);
  }
}