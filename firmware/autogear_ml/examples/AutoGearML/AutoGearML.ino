/*
 * AutoGearML -- integration example for the Shifter Pro prototype (Arduino Nano 33 BLE Sense Rev2).
 *
 * STATUS: reference integration, NOT tested on the bicycle. The AutoGear prototype is no longer
 * available; the library itself (src/) is verified against the Python reference by
 * firmware/host_test. Sensor calibration, servo angles and the gear ladder below are placeholders
 * to adapt to the bicycle.
 *
 * Once per second:
 *   cadence (Hall sensor) + slope (barometer) + acceleration (IMU) -> 14 inputs
 *   -> 100 boosted trees -> target development -> hysteresis controller -> at most one servo step
 *
 * Libraries: Arduino_BMI270_BMM150 (IMU), Arduino_LPS22HB (barometer), Servo, AutoGearML.
 */
#include <Arduino_BMI270_BMM150.h>
#include <Arduino_LPS22HB.h>
#include <Servo.h>
#include <autogear_ml.h>

/* ---------------- bicycle-specific calibration (placeholders) ---------------- */
static const uint8_t N_GEARS = 7;
/* development of each gear, metres per crank revolution, smallest first */
static const float DEVELOPMENT_M[N_GEARS] = {3.94f, 4.46f, 5.11f, 5.65f, 6.37f, 7.18f, 7.78f};
/* servo angle that engages each gear */
static const uint8_t SERVO_ANGLE[N_GEARS] = {20, 42, 64, 86, 108, 130, 152};
static const uint8_t HALL_PIN = 2;
static const uint8_t SERVO_PIN = 9;

/* ---------------- controller setting selected in the study ---------------- */
static const float H_MARGIN = 0.6f;      /* hysteresis, in steps */
static const uint8_t K_UP = 3;           /* persistence before an upshift, s */
static const uint8_t K_DOWN = 3;         /* 1 s makes downshifts faster (see the report, Sec. 5.4) */

static ag_features_t features;
static ag_controller_t controller;
static Servo servo;

static volatile uint32_t last_pulse_us = 0, pulse_period_us = 0;
static void on_crank_pulse() {
    uint32_t now = micros();
    pulse_period_us = now - last_pulse_us;
    last_pulse_us = now;
}

static float cadence_rpm() {
    noInterrupts();
    uint32_t period = pulse_period_us, last = last_pulse_us;
    interrupts();
    if (period == 0 || micros() - last > 4000000UL) return 0.0f;   /* no pulse for 4 s: stopped */
    return 60.0e6f / (float)period;
}

/* slope = altitude change over distance travelled, both over ~10 s */
static float altitude_hist[10], distance_hist[10];
static uint8_t hist_i = 0;
static float distance_m = 0.0f;

static float altitude_m() {
    float p_kpa = BARO.readPressure();
    return 44330.0f * (1.0f - powf(p_kpa / 101.325f, 0.1903f));
}

static float slope_estimate(float v_hat) {
    distance_m += v_hat;                     /* 1 s period */
    float alt = altitude_m();
    float dz = alt - altitude_hist[hist_i], dx = distance_m - distance_hist[hist_i];
    altitude_hist[hist_i] = alt;
    distance_hist[hist_i] = distance_m;
    hist_i = (hist_i + 1) % 10;
    float s = dx > 5.0f ? dz / dx : 0.0f;
    return s > 0.25f ? 0.25f : (s < -0.25f ? -0.25f : s);
}

/* longitudinal acceleration, gravity removed (x axis along the frame) */
static float longitudinal_acc(float slope) {
    float ax, ay, az;
    if (!IMU.accelerationAvailable()) return 0.0f;
    IMU.readAcceleration(ax, ay, az);        /* in g */
    float a = 9.81f * ax - 9.81f * slope / sqrtf(1.0f + slope * slope);
    return a > 3.0f ? 3.0f : (a < -3.0f ? -3.0f : a);
}

void setup() {
    IMU.begin();
    BARO.begin();
    pinMode(HALL_PIN, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(HALL_PIN), on_crank_pulse, FALLING);
    servo.attach(SERVO_PIN);
    ag_features_reset(&features);
    ag_controller_init(&controller, DEVELOPMENT_M, N_GEARS, 3, H_MARGIN, K_UP, K_DOWN, 0);
    servo.write(SERVO_ANGLE[controller.gear - 1]);
    float a0 = altitude_m();
    for (uint8_t i = 0; i < 10; ++i) { altitude_hist[i] = a0; distance_hist[i] = 0.0f; }
}

void loop() {
    static uint32_t next_ms = 0;
    if (millis() < next_ms) return;
    next_ms += 1000;

    float cadence = cadence_rpm();
    uint8_t coasting = cadence < 15.0f;
    static float slope = 0.0f;                                /* estimate of the previous second */
    float acc = longitudinal_acc(slope);
    float x[AG_N_INPUTS];
    float v_hat = ag_features_push(&features, cadence, ag_controller_development(&controller),
                                   slope, acc, coasting, x);
    slope = slope_estimate(v_hat);

    float target = ag_gbdt_predict(x);                         /* log metres per revolution */
    uint8_t pedalling = !coasting && cadence > AG_PEDAL_MIN_CADENCE_RPM;
    if (ag_controller_step(&controller, target, pedalling) != 0) {
        servo.write(SERVO_ANGLE[controller.gear - 1]);          /* one step, chain turning */
    }
}
