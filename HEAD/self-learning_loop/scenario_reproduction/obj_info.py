import csv
import re
import pickle as pkl

import numpy as np

# init dict to save vehicles & walkers info
obj_data = {
    'vehicles': {},
    'walkers': {}
}

with open('Obj_info_batch_2.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)


    for header in headers:
        match_vehicle = re.match(r'field\.vehicle_info_batch(\d+).*actor_pos\.x', header)
        if match_vehicle:
            vehicle_id = int(match_vehicle.group(1))  # extract vehicle ID
            if vehicle_id not in obj_data['vehicles']:
                obj_data['vehicles'][vehicle_id] = {
                    'position': [],
                    'velocity': [],
                    'psi': [],
                    'width': [],
                    'length': []
                }

        match_walker = re.match(r'field\.walker_info_batch(\d+).*actor_pos\.x', header)
        if match_walker:
            walker_id = int(match_walker.group(1))  # extract vehicle ID
            if walker_id not in obj_data['walkers']:
                obj_data['walkers'][walker_id] = {
                    'position': [],
                    'velocity': [],
                    'psi': [],
                    'width': [],
                    'length': []
                }


    for row in reader:

        for vehicle_id, vehicle_info in obj_data['vehicles'].items():
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.x')
            position_x = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.y')
            position_y = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_pos.z')
            position_z = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.x')
            velocity_x = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.y')
            velocity_y = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_vel.z')
            velocity_z = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_psi')
            psi = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_width')
            width = row[index] if index != -1 else None
            index = headers.index(f'field.vehicle_info_batch{vehicle_id}.actor_length')
            length = row[index] if index != -1 else None

            vehicle_info['position'].append((float(position_x), -float(position_y), 0))
            vehicle_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
            vehicle_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))
            # vehicle_info['psi'].append(float(psi))
            vehicle_info['width'].append(float(width))
            vehicle_info['length'].append(float(length))

        # 处理行人数据
        for walker_id, walker_info in obj_data['walkers'].items():
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.x')
            position_x = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.y')
            position_y = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_pos.z')
            position_z = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.x')
            velocity_x = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.y')
            velocity_y = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_vel.z')
            velocity_z = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_psi')
            psi = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_width')
            width = row[index] if index != -1 else None
            index = headers.index(f'field.walker_info_batch{walker_id}.actor_length')
            length = row[index] if index != -1 else None


            # walker_info['position'].append((abs(float(position_x)), abs(float(position_y)), float(position_z)))
            walker_info['position'].append((float(position_x), -float(position_y), 0))
            walker_info['velocity'].append((float(velocity_x), -float(velocity_y), float(velocity_z)))
            walker_info['psi'].append((2*np.pi-float(psi)) % (2*np.pi))
            walker_info['width'].append(float(width))
            walker_info['length'].append(float(length))


obj_data['vehicles'] = {k: v for k, v in obj_data['vehicles'].items() if not all(w == 0 for w in v['width'])}
obj_data['walkers'] = {k: v for k, v in obj_data['walkers'].items() if not all(w == 0 for w in v['width'])}
for vehicle_id, vehicle_info in obj_data['vehicles'].items():
    vehicle_info['position'] = np.array(vehicle_info['position'])
for walker_id, walker_info in obj_data['walkers'].items():
    walker_info['position'] = np.array(walker_info['position'])


# print("Updated vehicles data:", obj_data['vehicles'][2])
# print("Updated walkers data:", obj_data['walkers'][2])


# for vehicle_id, vehicle_info in obj_data['vehicles'].items():
#     print(f"Vehicle {vehicle_id} data: {vehicle_info}")
#
# for walker_id, walker_info in obj_data['walkers'].items():
#     print(f"Walker {walker_id} data: {walker_info}")

pickle_data = pkl.dumps(obj_data)

with open('obj_info.pkl', 'wb') as f: # save to pickle
    f.write(pickle_data)

