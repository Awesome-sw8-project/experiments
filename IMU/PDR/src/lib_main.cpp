#include <iostream>
#include <deadreckoner.h>
#include <chrono>

#define PERIOD_MS 20

// First way-point: x = 208.86206, y = 216.74796.
int main()
{
    auto period_s = std::chrono::duration_cast<std::chrono::duration<float, std::chrono::seconds::period>>(std::chrono::milliseconds(PERIOD_MS));
    DeadReckoner pdr;
    SensorReading* readings = new SensorReading[5];
    readings[0] = {.orientation = Quaternion(Vector(0.05849466, 0.033627603, -0.7411096)), .acceleration = Vector(-1.6574097, -0.03213501, 17.939987)};
    readings[1] = {.orientation = Quaternion(Vector(0.049627326, 0.030867666, -0.7375845)), .acceleration = Vector(-1.4868317, 0.35154724, 17.31868)};
    readings[2] = {.orientation = Quaternion(Vector(0.05261724, 0.026115067, -0.7339577)), .acceleration = Vector(-1.6203003, 0.512558, 15.494308)};
    readings[3] = {.orientation = Quaternion(Vector(0.05261724, 0.026115067, -0.7339577)), .acceleration = Vector(-1.7059021, 0.053466797, 14.442657)};
    readings[4] = {.orientation = Quaternion(Vector(0.05445788, 0.023243016, -0.7321772)), .acceleration = Vector(-1.8675079, -0.08720398, 12.749954)};

    std::cout << "Heading\t\tX\tY\tZ" << std::endl;

    for (unsigned i = 3; i <= 5; i++)
    {
        pdr.updatePositionAndHeadingBasedOnSensorData(readings, i, period_s.count());
        std::cout << pdr.currentHeading << "\t\t" << pdr.position.x << "\t" << pdr.position.y << "\t" << pdr.position.z << std::endl;
    }

    delete[] readings;
    return 0;
}
