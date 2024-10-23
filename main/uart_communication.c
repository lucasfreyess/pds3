#include "uart_communication.h"
#include "driver/uart.h"
#include "esp_log.h"
#include "string.h"

#define TX_PIN 17
#define RX_PIN 16
#define UART_PORT_NUM UART_NUM_2
#define BUF_SIZE 1024

void uart_init() {
    const uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
    };
    uart_param_config(UART_PORT_NUM, &uart_config);
    uart_set_pin(UART_PORT_NUM, TX_PIN, RX_PIN, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE);
    
    if (uart_is_driver_installed(UART_PORT_NUM)) {
        ESP_LOGI("UART", "UART driver is already installed.");
    } else {
        ESP_LOGI("UART", "UART driver is not installed.");
    }
    
    uart_driver_install(UART_PORT_NUM, BUF_SIZE * 2, 0, 0, NULL, 0);
}

void uart_send_data(const char* data) {
    uart_write_bytes(UART_PORT_NUM, data, strlen(data));
}

void uart_receive_data() {
    uint8_t data[BUF_SIZE];
    int len = uart_read_bytes(UART_PORT_NUM, data, BUF_SIZE, 20 / portTICK_PERIOD_MS);
    if (len > 0) {
        ESP_LOGI("UART_RX", "Received data: %s", data);
    }
}
