#include "cia417_mock.h"

#include <limits.h>
#include <string.h>

void cia417_init(struct cia417_od *od, uint8_t floor_count)
{
	memset(od, 0, sizeof(*od));
	od->op_mode = CIA417_OP_INIT;
	od->drive_state = CIA417_DRIVE_IDLE;
	od->floor_count = floor_count > CIA417_MAX_FLOORS ?
				  CIA417_MAX_FLOORS :
				  floor_count;
	od->car_position = 1;
}

bool cia417_register_call(struct cia417_od *od, uint8_t floor)
{
	if (floor == 0 || floor > od->floor_count) {
		return false;
	}
	od->hall_calls[floor] = 1;
	return true;
}

bool cia417_next_target(const struct cia417_od *od, uint8_t current_floor,
			uint8_t *target_out)
{
	uint8_t best_distance = UINT8_MAX;
	int best_floor = -1;

	for (uint8_t floor = 1; floor <= od->floor_count; ++floor) {
		if (od->hall_calls[floor] == 0 && od->car_calls[floor] == 0) {
			continue;
		}

		uint8_t distance = current_floor > floor ? current_floor - floor :
							   floor - current_floor;
		if (distance < best_distance) {
			best_distance = distance;
			best_floor = floor;
		}
	}

	if (best_floor < 0) {
		return false;
	}
	*target_out = (uint8_t)best_floor;
	return true;
}

void cia417_complete_call(struct cia417_od *od, uint8_t floor)
{
	if (floor == 0 || floor > od->floor_count) {
		return;
	}
	od->hall_calls[floor] = 0;
	od->car_calls[floor] = 0;
}

uint8_t cia417_pending_call_count(const struct cia417_od *od)
{
	uint8_t total = 0;
	for (uint8_t floor = 1; floor <= od->floor_count; ++floor) {
		if (od->hall_calls[floor] || od->car_calls[floor]) {
			total++;
		}
	}
	return total;
}

