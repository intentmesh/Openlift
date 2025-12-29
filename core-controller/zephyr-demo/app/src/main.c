#include <stdlib.h>
#include <string.h>

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/shell/shell.h>

#include "cia417_mock.h"

LOG_MODULE_REGISTER(split_core, LOG_LEVEL_INF);

enum safety_state {
	SAFETY_INIT = 0,
	SAFETY_ESTOP = 1,
	SAFETY_IDLE = 2,
	SAFETY_DRIVE_READY = 3,
};

struct safety_ctx {
	enum safety_state state;
	bool estop;
	bool door_closed;
	bool door_locked;
	bool drive_request;
	bool motion_active;
};

static struct safety_ctx g_safety = {
	.state = SAFETY_INIT,
	.estop = false,
	.door_closed = true,
	.door_locked = true,
	.drive_request = false,
	.motion_active = false,
};

static struct cia417_od g_od;

static K_MUTEX_DEFINE(safety_mutex);
static K_MUTEX_DEFINE(od_mutex);

static const char *safety_state_str(enum safety_state st)
{
	switch (st) {
	case SAFETY_INIT:
		return "Init";
	case SAFETY_ESTOP:
		return "Estop";
	case SAFETY_IDLE:
		return "Idle";
	case SAFETY_DRIVE_READY:
		return "DriveReady";
	default:
		return "Unknown";
	}
}

static void log_snapshot(void)
{
	k_mutex_lock(&safety_mutex, K_FOREVER);
	struct safety_ctx safety = g_safety;
	k_mutex_unlock(&safety_mutex);

	k_mutex_lock(&od_mutex, K_FOREVER);
	struct cia417_od od = g_od;
	uint8_t pending = cia417_pending_call_count(&g_od);
	k_mutex_unlock(&od_mutex);

	LOG_INF("Safety=%s estop=%d door_closed=%d lock=%d motion=%d floor=%u pending=%u door_cmd=0x%02x",
		safety_state_str(safety.state), safety.estop, safety.door_closed,
		safety.door_locked, safety.motion_active, od.car_position,
		pending, od.door_control);
}

static void safety_thread(void *, void *, void *)
{
	while (1) {
		enum safety_state prev;
		enum safety_state next;
		bool safe_inputs;

		k_mutex_lock(&safety_mutex, K_FOREVER);
		prev = g_safety.state;
		next = prev;
		safe_inputs = (!g_safety.estop && g_safety.door_closed &&
			       g_safety.door_locked);

		if (g_safety.estop) {
			next = SAFETY_ESTOP;
		} else if (next == SAFETY_INIT) {
			next = SAFETY_IDLE;
		} else if (!safe_inputs) {
			next = SAFETY_IDLE;
		} else if (next == SAFETY_IDLE && g_safety.drive_request) {
			next = SAFETY_DRIVE_READY;
			g_safety.drive_request = false;
		} else if (next == SAFETY_DRIVE_READY && !g_safety.motion_active) {
			/* drop to idle when motion complete */
			k_mutex_lock(&od_mutex, K_FOREVER);
			bool pending = cia417_pending_call_count(&g_od) > 0;
			k_mutex_unlock(&od_mutex);
			if (!pending) {
				next = SAFETY_IDLE;
			}
		}

		if (next != g_safety.state) {
			g_safety.state = next;
		}
		k_mutex_unlock(&safety_mutex);

		if (prev != next) {
			log_snapshot();
		}

		k_sleep(K_MSEC(50));
	}
}

static void motion_thread(void *, void *, void *)
{
	bool moving = false;
	uint8_t target_floor = 0;
	uint8_t dwell_ticks = 0;

	while (1) {
		enum safety_state state;
		bool estop_active;

		k_mutex_lock(&safety_mutex, K_FOREVER);
		state = g_safety.state;
		estop_active = g_safety.estop;
		k_mutex_unlock(&safety_mutex);

		if (estop_active || state == SAFETY_ESTOP) {
			k_sleep(K_MSEC(100));
			continue;
		}

		k_mutex_lock(&od_mutex, K_FOREVER);
		uint8_t current_floor = g_od.car_position;

		if (!moving) {
			/* Acquire a new target if safety will allow it */
			uint8_t next_target;
			if (cia417_next_target(&g_od, current_floor, &next_target)) {
				target_floor = next_target;

				k_mutex_lock(&safety_mutex, K_FOREVER);
				g_safety.drive_request = true;
				g_safety.motion_active = true;
				k_mutex_unlock(&safety_mutex);
				moving = true;
				g_od.door_control = CIA417_DOOR_CMD_CLOSE;
				LOG_INF("Dispatching car toward floor %u", target_floor);
			}
		} else if (state == SAFETY_DRIVE_READY) {
			if (current_floor == target_floor) {
				moving = false;
				dwell_ticks = 5;
				g_od.drive_state = CIA417_DRIVE_STOPPED;
				g_od.door_control = CIA417_DOOR_CMD_OPEN;
				cia417_complete_call(&g_od, target_floor);
				LOG_INF("Arrived at floor %u – opening door", target_floor);

				k_mutex_lock(&safety_mutex, K_FOREVER);
				g_safety.motion_active = false;
				k_mutex_unlock(&safety_mutex);
			} else {
				if (current_floor < target_floor) {
					g_od.car_position++;
					g_od.drive_state = CIA417_DRIVE_ACCEL;
				} else {
					g_od.car_position--;
					g_od.drive_state = CIA417_DRIVE_DECEL;
				}
				LOG_INF("Moving... floor=%u target=%u", g_od.car_position, target_floor);
			}
		}

		if (!moving && dwell_ticks > 0) {
			dwell_ticks--;
			if (dwell_ticks == 0) {
				g_od.door_control = CIA417_DOOR_CMD_CLOSE;
				LOG_INF("Door closed, returning to idle");
			}
		}

		k_mutex_unlock(&od_mutex);
		k_sleep(K_SECONDS(1));
	}
}

K_THREAD_DEFINE(safety_tid, 2048, safety_thread, NULL, NULL, NULL, 3, 0, 0);
K_THREAD_DEFINE(motion_tid, 2048, motion_thread, NULL, NULL, NULL, 4, 0, 0);

static int cmd_call(const struct shell *shell, size_t argc, char **argv)
{
	if (argc < 2) {
		shell_error(shell, "Usage: call <floor>");
		return -EINVAL;
	}

	uint8_t floor = (uint8_t)strtoul(argv[1], NULL, 10);

	k_mutex_lock(&od_mutex, K_FOREVER);
	bool accepted = cia417_register_call(&g_od, floor);
	k_mutex_unlock(&od_mutex);

	if (accepted) {
		shell_print(shell, "Registered hall call for floor %u", floor);
	} else {
		shell_error(shell, "Floor %u out of range (1-%u)", floor, g_od.floor_count);
	}
	return 0;
}

static int cmd_sensor(const struct shell *shell, size_t argc, char **argv)
{
	if (argc < 3) {
		shell_error(shell, "Usage: sensor <door|lock|estop> <value>");
		return -EINVAL;
	}

	const char *signal = argv[1];
	const char *value = argv[2];

	k_mutex_lock(&safety_mutex, K_FOREVER);
	if (strcmp(signal, "door") == 0) {
		g_safety.door_closed = (strcmp(value, "closed") == 0);
		shell_print(shell, "Door sensor => %s", g_safety.door_closed ? "closed" : "open");
	} else if (strcmp(signal, "lock") == 0) {
		g_safety.door_locked = (strcmp(value, "engaged") == 0);
		shell_print(shell, "Hoistway lock => %s",
			    g_safety.door_locked ? "engaged" : "released");
	} else if (strcmp(signal, "estop") == 0) {
		g_safety.estop = (strcmp(value, "trip") == 0);
		if (!g_safety.estop && g_safety.state == SAFETY_ESTOP) {
			g_safety.state = SAFETY_IDLE;
		}
		shell_print(shell, "Estop => %s", g_safety.estop ? "TRIPPED" : "RESET");
	} else {
		shell_error(shell, "Unknown sensor '%s'", signal);
		k_mutex_unlock(&safety_mutex);
		return -EINVAL;
	}
	k_mutex_unlock(&safety_mutex);
	log_snapshot();
	return 0;
}

static int cmd_status(const struct shell *shell, size_t argc, char **argv)
{
	ARG_UNUSED(argc);
	ARG_UNUSED(argv);

	k_mutex_lock(&safety_mutex, K_FOREVER);
	struct safety_ctx safety = g_safety;
	k_mutex_unlock(&safety_mutex);

	k_mutex_lock(&od_mutex, K_FOREVER);
	struct cia417_od od = g_od;
	uint8_t pending = cia417_pending_call_count(&g_od);
	k_mutex_unlock(&od_mutex);

	shell_print(shell,
		    "Safety=%s estop=%d door_closed=%d locked=%d motion=%d pending=%u floor=%u target_cmd=0x%02x",
		    safety_state_str(safety.state), safety.estop,
		    safety.door_closed, safety.door_locked, safety.motion_active,
		    pending, od.car_position, od.door_control);
	return 0;
}

SHELL_CMD_REGISTER(call, NULL, "Register a CiA-417 hall call", cmd_call);
SHELL_CMD_REGISTER(sensor, NULL,
		   "Set sensor state: sensor <door|lock|estop> <closed/open, engaged/released, trip/reset>",
		   cmd_sensor);
SHELL_CMD_REGISTER(status, NULL, "Print split-core status", cmd_status);

void main(void)
{
	cia417_init(&g_od, 16);
	g_od.op_mode = CIA417_OP_NORMAL;
	LOG_INF("OpenLift split-kernel demo ready. Use 'call <floor>' to enqueue traffic.");
	log_snapshot();
}

