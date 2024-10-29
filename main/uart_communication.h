#ifndef UART_COMMUNICATION_H
#define UART_COMMUNICATION_H

#ifdef __cplusplus
extern "C" {
#endif

void uart_init(void);
void uart_send_data(const char* data);
int uart_receive_data(void);

#ifdef __cplusplus
}
#endif

#endif // UART_COMMUNICATION_H
