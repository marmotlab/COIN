import copy

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.pgblock.ramp import InRampOnStraight
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MATIntersectionConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3)
        #-Road(TInterSection.node(1, 0, 0), TInterSection.node(1, 0, 1))
       # -Road(TInterSection.node(1, 1, 0), TInterSection.node(1, 1, 1))
    ],
    num_agents=30,
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120
)


class MATIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        # Build TIntersection
        TInterSection.EXIT_PART_LENGTH = length

        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}
        last_block = TInterSection(
            1,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )

        # if self.config["lane_num"] > 1:
        #     # We disable U turn in TinyInter environment!
        #     last_block.enable_u_turn(True)
        # else:
        #     last_block.enable_u_turn(False)

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)

        # Build Rampin
        InRampOnStraight.EXIT_PART_LENGTH = 20

        if "radius" in self.config and self.config["radius"]:
            extra_kwargs = dict(radius=self.config["radius"])
        else:
            extra_kwargs = {}

        last_block = InRampOnStraight(
            2,
            last_block.get_socket(index=0),
            self.road_network,
            random_seed=1,
            ignore_intersection_checking=False,
            **extra_kwargs
        )

        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)

class MATIntersectionSpawnManager(SpawnManager):
    def __init__(self, disable_u_turn=False):
        super(MATIntersectionSpawnManager, self).__init__()
        self.disable_u_turn = disable_u_turn

    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        if self.disable_u_turn:  # Remove the spawn road from end roads
            end_roads = [r for r in end_roads if Road(*vehicle_config["spawn_lane_index"][:2]) != r]
        end_road = -self.np_random.choice(end_roads)  # Use negative road!
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MATIntersectionPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MATIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentTIntersectionEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MATIntersectionConfig, allow_add_new_key=True)

    def setup_engine(self):
        disable_u_turn = self.config["map_config"]["lane_num"] < 2
        super(MultiAgentTIntersectionEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MATIntersectionPGMapManager())
        self.engine.update_manager("spawn_manager", MATIntersectionSpawnManager(disable_u_turn=disable_u_turn))



def _vis():
    env = MultiAgentTIntersectionEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            #
            "use_render": True,
            "debug": True,
            "allow_respawn": False,
            "manual_control": True,
            "num_agents": 8,
            "delay_done": 2,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        actions = {k: [0.0, 1.0] for k in env.agents.keys()}
        if len(env.agents) == 1:
            actions = {k: [-0, 1.0] for k in env.agents.keys()}
        o, r, tm, tc, info = env.step(actions)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        # render_text = {
        #     "total_r": total_r,
        #     "episode length": ep_s,
        #     "cam_x": env.main_camera.camera_x,
        #     "cam_y": env.main_camera.camera_y,
        #     "cam_z": env.main_camera.top_down_camera_height,
        #     "alive": len(env.agents)
        # }
        # env.render(text=render_text)
        # env.render(mode="top_down")
        if tm["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            env.reset()
            # break
        if len(env.agents) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


if __name__ == "__main__":

    _vis()
