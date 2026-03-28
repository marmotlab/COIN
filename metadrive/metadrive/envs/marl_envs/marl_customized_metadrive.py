import copy
from math import floor
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.std_intersection import StdInterSection
from metadrive.component.pgblock.std_t_intersection import StdTInterSection
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.straight import Straight
from metadrive.component.road_network.road import Road
from metadrive.policy.idm_policy import ManualControllableIDMPolicy

from metadrive.manager.spawn_manager import SpawnManager

from metadrive.utils import Config

CUSTOMIZED_MULTI_AGENT_METADRIVE_DEFAULT_CONFIG = dict()


class CustomizedMultiAgentSpawnManager(SpawnManager):
    def __init__(self, disable_u_turn=False):
        super(SpawnManager, self).__init__()
        self.initialized = True
        self.num_agents = self.engine.global_config["num_agents"]
        self.exit_length = (self.engine.global_config["map_config"]["exit_length"] - FirstPGBlock.ENTRANCE_LENGTH)
        self.num_slots = int(floor(self.exit_length / SpawnManager.RESPAWN_REGION_LONGITUDE))
        assert self.exit_length >= self.RESPAWN_REGION_LONGITUDE, (
            "The exist length {} should greater than minimal longitude interval {}.".format(
                self.exit_length, self.RESPAWN_REGION_LONGITUDE
            )
        )
        self.lane_num = self.engine.global_config["map_config"]["lane_num"]
        self.spawn_roads = []
        self.safe_spawn_places = {}
        self.need_update_spawn_places = True
        self.spawn_places_used = []  # reset every step

        agent_configs = copy.copy(self.engine.global_config["agent_configs"])
        self._init_agent_configs = agent_configs
        self.disable_u_turn = disable_u_turn

    def reset(self):
        # reset spawn roads according to the randomly generated map
        spawn_roads = [Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)]

        # get map blocks
        blocks = self.engine.current_map.blocks.copy()

        # pop out the first PG block
        blocks.pop(0)
        block_num = len(blocks)

        # add new spawn roads
        for i in range(block_num):
            # all spawn roads are sockets and negative roads (except for First PGBlock)

            # get sockets of a block
            socket_list = blocks[i].get_socket_list()

            if i < block_num - 1:
                # remove the socket used by next block
                used_socket = blocks[i + 1].pre_block_socket
                socket_list.remove(used_socket)

            # add spawn road
            selected_spawn_road = [spawn_socket.negative_road for spawn_socket in socket_list]
            spawn_roads.extend(selected_spawn_road)

        if self.num_agents is not None:
            assert self.num_agents > 0 or self.num_agents == -1
            print("max capacity of the map is: ", self.max_capacity(
                spawn_roads, self.exit_length + FirstPGBlock.ENTRANCE_LENGTH, self.lane_num
            ))
            assert self.num_agents <= self.max_capacity(
                spawn_roads, self.exit_length + FirstPGBlock.ENTRANCE_LENGTH, self.lane_num
            ), (
                "Too many agents! We only accept {} agents, but you have {} agents!".format(
                    self.lane_num * len(spawn_roads) * self.num_slots, self.num_agents
                )
            )

        interval = self.exit_length / self.num_slots
        self._longitude_spawn_interval = interval

        available_agent_configs, safe_spawn_places = self._auto_fill_spawn_roads_randomly(spawn_roads)
        self.available_agent_configs = available_agent_configs
        self.safe_spawn_places = {place["identifier"]: place for place in safe_spawn_places}
        self.spawn_roads = spawn_roads
        self.engine.global_config.spawn_roads = spawn_roads
        super(CustomizedMultiAgentSpawnManager, self).reset()

    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        if self.disable_u_turn:  # Remove the spawn road from end roads
            end_roads = [r for r in end_roads if Road(*vehicle_config["spawn_lane_index"][:2]) != r]
        end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class CustomizedMultiAgentEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(CUSTOMIZED_MULTI_AGENT_METADRIVE_DEFAULT_CONFIG,
                                                           allow_add_new_key=True)

    def setup_engine(self):
        disable_u_turn = self.config["map_config"]["lane_num"] < 2
        super(MultiAgentMetaDrive, self).setup_engine()
        self.engine.register_manager("spawn_manager", CustomizedMultiAgentSpawnManager(disable_u_turn=disable_u_turn))


if __name__ == "__main__":
    env = CustomizedMultiAgentEnv(
        {
            "start_seed": 234,
            "num_scenarios": 10000,
            "use_render": True,
            # "manual_control": True,
            "num_agents": 30,
            "map": 7,
            "allow_respawn": True,
            "crash_done": False,
            "agent_policy": ManualControllableIDMPolicy,
        }
    )

    env.reset()
    for i in range(100000):
        o, r, tm, tc, info = env.step({v_id: [0, 0] for v_id in env.agents.keys()})
        # if i % 50 == 0:
        # print("next reset seed:",200+i)

        # env.reset(seed = 200+i)
        if tm["__all__"]:
            env.reset()
    env.close()
