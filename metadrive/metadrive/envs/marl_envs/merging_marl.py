import copy
from math import floor

from metadrive.component.map.pg_map import PGMap
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.road_network import Road
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.manager.pg_map_manager import PGMapManager
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.utils import Config

MAMergingConfig = dict(
    spawn_roads=[
        # Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        Road(InRampOnStraight.node(1, 0, 1), InRampOnStraight.node(1, 0, 2)),
    ],
    
    num_agents=10,
    map_config=dict(exit_length=60, lane_num=2),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120
)


class MAMergingMap(PGMap):
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

        # Build Inramp
        last_block = InRampOnStraight(
            1, 
            last_block.get_socket(index=0), 
            self.road_network, 
            random_seed=1, 
            ignore_intersection_checking=False,
        )
        
        last_block.construct_block(
            parent_node_path,
            physics_world,
        )
        self.blocks.append(last_block)
        
        # Build Outramp
        last_block = OutRampOnStraight(
            2, 
            last_block.get_socket(index=0), 
            self.road_network, 
            random_seed=1, 
            ignore_intersection_checking=False,
        )
        
        last_block.construct_block(
            parent_node_path,
            physics_world,
        )
        self.blocks.append(last_block)

class MAMergingSpawnManager(SpawnManager):
    def __init__(self):
        super(MAMergingSpawnManager, self).__init__()
        
    def _auto_fill_spawn_roads_randomly(self, spawn_roads):
        """
        Modified spawn lane selection

        """
        # print('还得是继承啊老哥们')
        num_slots = int(floor(self.exit_length / SpawnManager.RESPAWN_REGION_LONGITUDE))
        interval = self.exit_length / num_slots
        self._longitude_spawn_interval = interval
        if self.num_agents is not None:
            assert self.num_agents > 0 or self.num_agents == -1
            assert self.num_agents <= self.max_capacity(
                spawn_roads, self.exit_length + FirstPGBlock.ENTRANCE_LENGTH, self.lane_num
            ), (
                "Too many agents! We only accept {} agents, but you have {} agents!".format(
                    self.lane_num * len(spawn_roads) * num_slots, self.num_agents
                )
            )

        # We can spawn agents in the middle of road at the initial time, but when some vehicles need to be respawn,
        # then we have to set it to the farthest places to ensure safety (otherwise the new vehicles may suddenly
        # appear at the middle of the road!)
        agent_configs = []
        safe_spawn_places = []
        for i, road in enumerate(spawn_roads):
            # print(road.start_node, type(road.start_node))
            if not 'r' in road.start_node:
                for lane_idx in range(self.lane_num):
                    for j in range(num_slots):
                        long = 1 / 2 * self.RESPAWN_REGION_LONGITUDE + j * self.RESPAWN_REGION_LONGITUDE
                        lane_tuple = road.lane_index(lane_idx)  # like (>>>, 1C0_0_, 1) and so on.
                        agent_configs.append(
                            Config(
                                dict(
                                    identifier="|".join((str(s) for s in lane_tuple + (j, ))),
                                    config={
                                        "spawn_lane_index": lane_tuple,
                                        "spawn_longitude": long,
                                        "spawn_lateral": 0
                                    },
                                ),
                                unchangeable=True
                            )
                        )  # lock the spawn positions
                        if j == 0:
                            safe_spawn_places.append(copy.deepcopy(agent_configs[-1]))
            else:
                for lane_idx in range(1):
                    for j in range(3):
                        long = 1 / 2 * self.RESPAWN_REGION_LONGITUDE + j * self.RESPAWN_REGION_LONGITUDE
                        lane_tuple = road.lane_index(lane_idx)  # like (>>>, 1C0_0_, 1) and so on.
                        agent_configs.append(
                            Config(
                                dict(
                                    identifier="|".join((str(s) for s in lane_tuple + (j, ))),
                                    config={
                                        "spawn_lane_index": lane_tuple,
                                        "spawn_longitude": long,
                                        "spawn_lateral": 0
                                    },
                                ),
                                unchangeable=True
                            )
                        )  # lock the spawn positions
                        if j == 0:
                            safe_spawn_places.append(copy.deepcopy(agent_configs[-1]))

        return agent_configs, safe_spawn_places
    
    def update_destination_for(self, agent_id, vehicle_config):
        # end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        # for temp in end_roads:
        #     print(temp, type(temp))
        #     print(temp.start_node, type(temp.start_node))
        end_roads = [
            # Road(OutRampOnStraight.node(2, 0, 0), OutRampOnStraight.node(2, 0, 1)),
            Road(OutRampOnStraight.node(2, 1, 3), OutRampOnStraight.node(2, 1, 4)),
        ]
        end_road = self.np_random.choice(end_roads)
        vehicle_config["destination"] = end_road.end_node
        return vehicle_config


class MAMergingPGMapManager(PGMapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MAMergingMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentMergingEnv(MultiAgentMetaDrive):
    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MAMergingConfig, allow_add_new_key=True)

    def setup_engine(self):
        super(MultiAgentMergingEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MAMergingPGMapManager())
        self.engine.update_manager("spawn_manager", MAMergingSpawnManager())

def _vis():
    from metadrive.policy.idm_policy import IDMPolicy
    env = MultiAgentMergingEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": True,
            },
            #
            "use_render": True,
            "debug": True,
            "allow_respawn": True,
            "manual_control": True,
            "num_agents": 1,
            "delay_done": 2,
        }
    )
    o, _ = env.reset()
    total_r = 0
    ep_s = 0
    print(env.agent_manager.active_agents['agent0'].config['destination'])
    for i in range(1, 100000):
        actions={}
        for k, v in env.agents.items():
            policy = IDMPolicy(v, random_seed=1)
            action = policy.act()
            actions[k] = action
        # actions = {k: IDMPolicy(v) for k, v in env.agents.items()}
        if len(env.agents) == 1:
            actions = {k: IDMPolicy(v, random_seed=1).act() for k, v in env.agents.items()}
        o, r, tm, tc, info = env.step(actions)
        # env.render(mode="top_down", num_stack=25, camera_position=(135, 5), scaling=2, screen_record=False, window=True)

        for agent in list(o.keys()):
            if tm[agent]:
                print(info[agent]['arrive_dest'])
                
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
   
