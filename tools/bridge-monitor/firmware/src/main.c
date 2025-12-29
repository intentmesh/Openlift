#include <zephyr/device.h>
#include <zephyr/drivers/can.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/logging/log.h>
#include <zephyr/net/mqtt.h>
#include <zephyr/net/socket.h>
#include <zephyr/kernel.h>
#include <stdio.h>

LOG_MODULE_REGISTER(openlift_bridge, LOG_LEVEL_INF);

#define CAN_NODE DT_ALIAS(can_bridge)
#define DOOR_GPIO_NODE DT_ALIAS(door_closed)

static const struct device *const can_dev = DEVICE_DT_GET(CAN_NODE);
static const struct device *const door_gpio = DEVICE_DT_GET(DOOR_GPIO_NODE);

static struct gpio_callback door_cb;
static atomic_t door_state = ATOMIC_INIT(0);

static void door_isr(const struct device *dev, struct gpio_callback *cb, uint32_t pins)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(cb);
	ARG_UNUSED(pins);
	atomic_set(&door_state, gpio_pin_get(door_gpio, DT_GPIO_PIN(DOOR_GPIO_NODE, gpios)));
}

static void publish_json(const char *json)
{
	printk("%s\n", json);
	/* MQTT publish hook would go here */
}

static void door_task(void)
{
	bool last_state = false;
	while (true) {
		bool current = atomic_get(&door_state);
		if (current != last_state) {
			char buf[128];
			snprintk(buf, sizeof(buf),
				 "{\"ts\":%lld,\"door_closed\":%s}",
				 k_uptime_get() / 1000, current ? "true" : "false");
			publish_json(buf);
			last_state = current;
		}
		k_sleep(K_MSEC(200));
	}
}

static void can_rx_handler(const struct device *dev, struct can_frame *frame, void *user_data)
{
	ARG_UNUSED(dev);
	ARG_UNUSED(user_data);
	char payload[32];
	for (int i = 0; i < frame->dlc; i++) {
		sprintf(payload + (i * 2), "%02X", frame->data[i]);
	}
	char buf[256];
	snprintk(buf, sizeof(buf),
		 "{\"ts\":%lld,\"can\":{\"id\":%u,\"dlc\":%d,\"data\":\"%s\"}}",
		 k_uptime_get() / 1000,
		 frame->id, frame->dlc, payload);
	publish_json(buf);
}

int main(void)
{
	if (!device_is_ready(can_dev)) {
		LOG_ERR("CAN device not ready");
		return 0;
}
	can_set_mode(can_dev, CAN_MODE_NORMAL);
	can_start(can_dev);
	static struct can_filter all_filter = {
		.id = 0,
		.mask = 0,
		.flags = 0,
	};
	can_add_rx_filter(can_dev, can_rx_handler, NULL, &all_filter);

	gpio_pin_configure(door_gpio, DT_GPIO_PIN(DOOR_GPIO_NODE, gpios),
			   GPIO_INPUT | DT_GPIO_FLAGS(DOOR_GPIO_NODE, gpios));
	gpio_pin_interrupt_configure(door_gpio, DT_GPIO_PIN(DOOR_GPIO_NODE, gpios),
				     GPIO_INT_EDGE_BOTH);
	gpio_init_callback(&door_cb, door_isr, BIT(DT_GPIO_PIN(DOOR_GPIO_NODE, gpios)));
	gpio_add_callback(door_gpio, &door_cb);

	k_thread_create(&(struct k_thread){0}, K_THREAD_STACK_DEFINE(stack, 1024), 1024,
			(k_thread_entry_t)door_task, NULL, NULL, NULL,
			K_PRIO_PREEMPT(7), 0, K_NO_WAIT);

	while (true) {
		k_sleep(K_SECONDS(1));
	}
}

