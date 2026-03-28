import math
import numpy as np

from metadrive.component.lane.straight_lane import StraightLane
from metadrive.component.pgblock.create_pg_block_utils import ExtendStraightLane, CreateRoadFrom, CreateAdverseRoad, \
    create_bend_straight, create_extension
from metadrive.component.pgblock.pg_block import PGBlock
from metadrive.component.lane.extension_lane import ExtensionDirection
from metadrive.component.road_network import Road
from metadrive.constants import Decoration, PGLineType
from metadrive.utils.pg.utils import check_lane_on_road
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace


class Ramp(PGBlock):
    """
                    InRamp                                             OutRamp

     start ----------- end ------------------           start ----------- end ------------------
     start ----------- end ------------------           start ----------- end ------------------
    (start ----------- end)[----------------]           start ----------- end [----------------]
                       end -----------------}          (start ---------- {end)
                      //                                                      \\
    { ---------------//                                                        \\---------------}
    """
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.RAMP_PARAMETER)
    SOCKET_NUM = 1

    # can be added to parameter space in the future
    RADIUS = 40  # meters
    ANGLE = 10  # degree
    LANE_TYPE = (PGLineType.CONTINUOUS, PGLineType.CONTINUOUS)
    SPEED_LIMIT = 12  # 12 m/s ~= 40 km/h
    CONNECT_PART_LEN = 20
    RAMP_LEN = 15


class CustomizedRamp(Ramp):
    ID = "M"
    EXTRA_PART = 10
    SOCKET_LEN = 20

    def _get_in_merge_part(self, att_lane: StraightLane, length: float):
        start = att_lane.end
        merge_part = create_extension(start, length, ExtensionDirection.SHRINK, width=self.lane_width)
        return merge_part

    def _get_out_merge_part(self, att_lane: StraightLane, length: float):
        start = att_lane.position(0, self.lane_width)
        merge_part = create_extension(start, length, ExtensionDirection.EXTEND, width=self.lane_width)
        return merge_part

    def _try_plug_into_previous_block(self) -> bool:
        length = self.get_config()[Parameter.length]
        extension_length = self.get_config()[Parameter.extension_length]
        no_cross = True

        # extend road part 0 road 0
        self.set_part_idx(0)
        sin_angle = math.sin(np.deg2rad(self.ANGLE))
        cos_angle = math.cos(np.deg2rad(self.ANGLE))
        longitude_len = sin_angle * self.RADIUS * 2 + cos_angle * self.CONNECT_PART_LEN + self.RAMP_LEN

        extend_lane = ExtendStraightLane(
            self.positive_basic_lane, longitude_len + self.EXTRA_PART, [PGLineType.BROKEN, PGLineType.CONTINUOUS]
        )
        extend_road = Road(self.pre_block_socket.positive_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane,
            self.positive_lane_num,
            extend_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        extend_road.get_lanes(self.block_network)[-1].line_types = [
            PGLineType.BROKEN if self.positive_lane_num != 1 else PGLineType.CONTINUOUS, PGLineType.CONTINUOUS
        ]

        # main acc part part 0 road 1
        acc_side_lane = ExtendStraightLane(
            extend_lane, length + extension_length, [extend_lane.line_types[0], PGLineType.CONTINUOUS]
        )
        acc_road = Road(extend_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            acc_side_lane,
            self.positive_lane_num,
            acc_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        left_line_type = PGLineType.CONTINUOUS if self.positive_lane_num == 1 else PGLineType.BROKEN
        acc_road.get_lanes(self.block_network)[-1].line_types = [left_line_type, PGLineType.BROKEN]
        #
        # ramp part, part 1
        self.set_part_idx(1)
        lateral_dist = (1 - cos_angle) * self.RADIUS * 2 + sin_angle * self.CONNECT_PART_LEN
        end_point = extend_lane.position(self.EXTRA_PART + self.RAMP_LEN, lateral_dist + self.lane_width)
        start_point = extend_lane.position(self.EXTRA_PART, lateral_dist + self.lane_width)
        straight_part = StraightLane(
            start_point, end_point, self.lane_width, self.LANE_TYPE, speed_limit=self.SPEED_LIMIT
        )
        straight_road = Road(self.add_road_node(), self.add_road_node())
        self.block_network.add_lane(straight_road.start_node, straight_road.end_node, straight_part)
        no_cross = (
                       not check_lane_on_road(
                           self._global_network,
                           straight_part,
                           0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross
        self.add_respawn_roads(straight_road)

        # p1 road 0, 1
        bend_1, connect_part = create_bend_straight(
            straight_part,
            self.CONNECT_PART_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            False,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_1_road = Road(straight_road.end_node, self.add_road_node())
        connect_road = Road(bend_1_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_1_road.start_node, bend_1_road.end_node, bend_1)
        self.block_network.add_lane(connect_road.start_node, connect_road.end_node, connect_part)
        no_cross = (
                       not check_lane_on_road(
                           self._global_network, bend_1, 0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross
        no_cross = (
                       not check_lane_on_road(
                           self._global_network,
                           connect_part,
                           0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross

        # p1, road 2, 3
        bend_2, acc_lane = create_bend_straight(
            connect_part,
            20,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            True,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        acc_lane.line_types = [PGLineType.BROKEN, PGLineType.CONTINUOUS]
        bend_2_road = Road(connect_road.end_node, self.road_node(0, 0))  # end at part1 road 0, extend road
        self.block_network.add_lane(bend_2_road.start_node, bend_2_road.end_node, bend_2)
        self.block_network.add_lane(acc_road.start_node, acc_road.end_node, acc_lane)
        no_cross = (
                       not check_lane_on_road(
                           self._global_network, bend_2, 0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross
        no_cross = (
                       not check_lane_on_road(
                           self._global_network, acc_lane, 0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross

        # p1, road 4, small circular to decorate
        merge_lane = self._get_in_merge_part(acc_lane, extension_length)
        merge_lane.line_types=[PGLineType.BROKEN, PGLineType.CONTINUOUS]
        self.block_network.add_lane(Decoration.start, Decoration.end, merge_lane)

        # extension part, part 0 road 2
        extend_lane_2 = ExtendStraightLane(
                        acc_side_lane, extension_length, [acc_side_lane.line_types[0], PGLineType.CONTINUOUS]
                    )
        extend_road_2 = Road(acc_road.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane_2,
            self.positive_lane_num,
            extend_road_2,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        extend_road_2.get_lanes(self.block_network)[-1].line_types = [PGLineType.BROKEN,PGLineType.CONTINUOUS]

        self.set_part_idx(2)
        # part 2 road 0
        straight_lane = ExtendStraightLane(
            extend_lane_2, length + extension_length,
            [extend_lane_2.line_types[0], PGLineType.CONTINUOUS]
        )
        acc_road_2 = Road(extend_road_2.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            straight_lane,
            self.positive_lane_num,
            acc_road_2,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        # part 3 road 0
        self.set_part_idx(3)
        extend_lane_3 = ExtendStraightLane(
            straight_lane, length + extension_length, [straight_lane.line_types[0], PGLineType.CONTINUOUS]
        )
        extend_road_3 = Road(acc_road_2.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            extend_lane_3,
            self.positive_lane_num,
            extend_road_3,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross
        _right_lane = extend_road_3.get_lanes(self.block_network)[-1]
        _left_line_type = PGLineType.CONTINUOUS if self.positive_lane_num == 1 else PGLineType.BROKEN
        _right_lane.line_types = [_left_line_type, PGLineType.CONTINUOUS]

        # part 3 road 1
        socket_lane = ExtendStraightLane(
            extend_lane_3, length * 2, [PGLineType.BROKEN, PGLineType.CONTINUOUS]
        )
        socket_road = Road(extend_road_3.end_node, self.add_road_node())
        no_cross = CreateRoadFrom(
            socket_lane,
            self.positive_lane_num,
            socket_road,
            self.block_network,
            self._global_network,
            side_lane_line_type=PGLineType.CONTINUOUS,
            ignore_intersection_checking=self.ignore_intersection_checking
        ) and no_cross

        self.add_sockets(self.create_socket_from_positive_road(socket_road))

        # part 4 road 0
        self.set_part_idx(4)
        merge_part_lane = self._get_out_merge_part(_right_lane, extension_length)
        merge_part_lane.line_types=[PGLineType.BROKEN, PGLineType.CONTINUOUS]
        self.block_network.add_lane("kongke", "suibian", merge_part_lane)

        deacc_lane_end = _right_lane.position(_right_lane.length, self.lane_width)
        deacc_lane = StraightLane(
            merge_part_lane.end, deacc_lane_end, self.lane_width, (PGLineType.BROKEN, PGLineType.CONTINUOUS)
        )
        self.block_network.add_lane(extend_road_3.start_node, extend_road_3.end_node, deacc_lane)
        no_cross = (
            not check_lane_on_road(
                self._global_network,
                deacc_lane,
                0.95,
                ignore_intersection_checking=self.ignore_intersection_checking
            )
        ) and no_cross

        bend_1, connect_part = create_bend_straight(
            deacc_lane,
            self.CONNECT_PART_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            True,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_1_road = Road(extend_road_3.end_node, self.add_road_node())
        connect_road = Road(bend_1_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_1_road.start_node, bend_1_road.end_node, bend_1)
        self.block_network.add_lane(connect_road.start_node, connect_road.end_node, connect_part)
        no_cross = (
                       not check_lane_on_road(
                           self._global_network, bend_1, 0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross
        no_cross = (
                       not check_lane_on_road(
                           self._global_network,
                           connect_part,
                           0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross

        bend_2, straight_part = create_bend_straight(
            connect_part,
            self.RAMP_LEN,
            self.RADIUS,
            np.deg2rad(self.ANGLE),
            False,
            self.lane_width,
            self.LANE_TYPE,
            speed_limit=self.SPEED_LIMIT
        )
        bend_2_road = Road(connect_road.end_node, self.add_road_node())
        straight_road = Road(bend_2_road.end_node, self.add_road_node())
        self.block_network.add_lane(bend_2_road.start_node, bend_2_road.end_node, bend_2)
        self.block_network.add_lane(straight_road.start_node, straight_road.end_node, straight_part)
        no_cross = (
                       not check_lane_on_road(
                           self._global_network, bend_2, 0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross
        no_cross = (
                       not check_lane_on_road(
                           self._global_network,
                           straight_part,
                           0.95,
                           ignore_intersection_checking=self.ignore_intersection_checking
                       )
                   ) and no_cross

        return no_cross

    def get_intermediate_spawn_lanes(self):
        """
        Remove lanes on socket
        """
        spawn_lanes = super(CustomizedRamp, self).get_intermediate_spawn_lanes()
        lane_on_socket = self.get_socket(0).get_positive_lanes(self.block_network)[0]
        filtered = []
        for lanes in spawn_lanes:
            if lane_on_socket in lanes:
                continue
            else:
                filtered.append(lanes)
        spawn_lanes = filtered
        assert sum([abs(l.length - CustomizedRamp.RAMP_LEN) <= 0.1 for ls in spawn_lanes for l in ls]) == 1
        return spawn_lanes



