#pragma once

#include <vector>
#include "problem_instance.h"

/*
 * This performs a 2-opt move by flipping a section of the route.
 * The boundaries of the section are given by first and last.
 * 
 * It may happen that the section is very long compared to the remaining part
 * of the route. In such case, the remaining part is flipped, to speed things
 * up as the result of such flip results in equivalent solution.
 *
 * The positions of the nodes inside the route are also updated to match
 * the order after the flip.
 */
void flip_route_section(std::vector<uint32_t> &route,
                        std::vector<uint32_t> &pos_in_route,
                        int32_t first, int32_t last);

/**
 * This is an implementation of an approximate 2-opt heuristic which uses the
 * nearest neighbor lists to limit the search for an improving move.
 */
int64_t two_opt_nn(const ProblemInstance &instance,
                   std::vector<uint32_t> &route,
                   bool use_dont_look_bits,
                   uint32_t nn_count);

int64_t two_opt_nn(const ProblemInstance &instance,
                   std::vector<uint32_t> &route,
                   std::vector<uint32_t> &check_queue,
                   uint32_t nn_count);

/*
 * Impl. of the 3-opt heuristic. Tries to change the order of nodes in
 * solution to shorten the travel distance.
 *
 * During the search for an improvement only edges connecting nn_count nearest
 * neighbors are taken into account to cut the overall search time.
 *
 * The solution's route is modified only if a better order was found.
 *
 * Function returns improvement over the previous route length (travel
 * distance).
 *
 * This implementation is based on the ideas proposed in:
 * Bentley, Jon Jouis. "Fast algorithms for geometric traveling
 * salesman problems." ORSA Journal on computing 4.4 (1992): 387-411.
 *
 * Returns a number of changes (moves) applied to the route.
*/
int64_t three_opt_nn(const ProblemInstance &instance,
                     std::vector<uint32_t> &sol,
                     bool use_dont_look_bits,
                     uint32_t nn_count);
