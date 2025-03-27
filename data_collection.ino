

#include <Wire.h>
#include "ICM20600.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
ICM20600 icm20600(true);

// Define analog pins
const int flexPins[] = {A0, A1, A2, A3, A4}; // 5 flex sensors

int flexValues[5];
float ax, ay, az;
float gx, gy, gz;

bool startLogging = true;

void setup() {
    
    Serial.begin(115200);
    Wire.begin();
    icm20600.initialize();
}

void loop() {

    // Read IMU sensor data
    ax = icm20600.getAccelerationX();
    ay = icm20600.getAccelerationY();
    az = icm20600.getAccelerationZ();
    gx = icm20600.getGyroscopeX();
    gy = icm20600.getGyroscopeY();
    gz = icm20600.getGyroscopeZ();

    // Read flex sensors
    for (int i = 0; i < 5; i++) {
        flexValues[i] = analogRead(flexPins[i]);
    }

    // **Use `sprintf` to send data in batch**
    char buffer[100];
    sprintf(buffer, "%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d,%d,%d\n",
            ax, ay, az, gx, gy, gz,
            flexValues[0], flexValues[1], flexValues[2], flexValues[3], flexValues[4]);
    Serial.print(buffer);
    delay(20);
    // Serial.flush();  // Force flush serial buffer
}
