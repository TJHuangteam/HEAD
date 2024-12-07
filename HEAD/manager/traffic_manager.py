from metadrive.component.vehicle.vehicle_type import DefaultVehicle
from metadrive.policy.idm_policy import IDMPolicy

class traffic_manager:
    def __init__(self, env):
        self.env = env
        self.actors_batch = []

    def setup_adv_traffic_vehicles(self):
        """
        Setup additional traffic vehicles in the environment.

        :param env: The environment instance
        :param num_vehicles: Number of additional vehicles to create
        :return: List of vehicle IDs
        grid world indices:
            row   0   1  2    3   4   5   6...       <== column
                  ----------------
             3 |  3   7  11  15  19  23  27...
             2 |  2   6  10  14  18  22  26...
             1 |  1   5  9   13  17  21  25...
             0 |  0   4  8   12  16  20  24...
             grid width = Lane width = 3.5 (m)
             grid length = 10 (m)
             ego col = 2
             ego row/lane = randomly initialized
        """
        ego_s = 5  # Ego vehicle's s position
        traffic_vehicle = []
        cfg = self.env.config["vehicle_config"]
        cfg["agent_policy"] = IDMPolicy  # Default to be IDM, but you can change it to other policies, e.g., RL policy
        grid_choices = [8, 11, 18, 20, 23]  #慢点修改一下

        for idx in grid_choices:
            col = idx // 4
            lane = idx - col * 4  # lane number [0, 3]
            s = ego_s + col * 10 - 20
            d = lane * 3.5
            position = [s, d]
            heading = 0.0  # Example heading, adjust as needed
            vehicle = self.env.engine.spawn_object(DefaultVehicle, vehicle_config=cfg, position=position, heading=heading)
            policy = IDMPolicy(vehicle, self.env.current_seed)
            traffic_vehicle.append(vehicle)
            self.actors_batch.append({
                'Actor': vehicle, 'Policy': policy, 'ID': vehicle.id
            })
        return [v.id for v in traffic_vehicle]
    def tick(self):
        for actor in self.actors_batch:
            actor['Actor'].before_step(actor['Policy'].act())
    def reset(self):
        for actor in self.actors_batch:
            self.env.engine.clear_objects(actor['ID'], True)
        self.actors_batch.clear()