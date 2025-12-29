#pragma once

#include <stdbool.h>
#include <stdint.h>

#define CIA417_MAX_FLOORS 64

#include <zephyr/sys/bitops.h>

#define CIA417_DOOR_CMD_OPEN  BIT(0)
#define CIA417_DOOR_CMD_CLOSE BIT(1)
#define CIA417_DOOR_CMD_NUDGE BIT(2)

enum cia417_op_mode {
	CIA417_OP_INIT = 0,
	CIA417_OP_NORMAL = 1,
	CIA417_OP_INSPECTION = 2,
	CIA417_OP_FIRE = 3,
	CIA417_OP_OUT_OF_SERVICE = 128,
};

enum cia417_drive_state {
	CIA417_DRIVE_UNKNOWN = 0,
	CIA417_DRIVE_IDLE = 1,
	CIA417_DRIVE_ACCEL = 2,
	CIA417_DRIVE_CONSTANT = 3,
	CIA417_DRIVE_DECEL = 4,
	CIA417_DRIVE_LEVELING = 5,
	CIA417_DRIVE_STOPPED = 6,
};

struct cia417_od {
	enum cia417_op_mode op_mode;
	enum cia417_drive_state drive_state;
	uint8_t floor_count;
	uint8_t car_position;
	uint8_t hall_calls[CIA417_MAX_FLOORS];
	uint8_t car_calls[CIA417_MAX_FLOORS];
	uint8_t door_control;
};

void cia417_init(struct cia417_od *od, uint8_t floor_count);
bool cia417_register_call(struct cia417_od *od, uint8_t floor);
bool cia417_next_target(const struct cia417_od *od, uint8_t current_floor,
			uint8_t *target_out);
void cia417_complete_call(struct cia417_od *od, uint8_t floor);
uint8_t cia417_pending_call_count(const struct cia417_od *od);

