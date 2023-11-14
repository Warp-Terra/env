import os
import time
import random
import math
import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
import cv2
import copy
import random
import pybullet_data

BEV_PIXEL_WIDTH =160
dir_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(dir_path + "/imgs/"):
    os.mkdir(dir_path + "/imgs/")

class BoxWorld(gym.Env):
    def __init__(self, Env_type="drop", isGUI=False, is_save_images=False, max_steps_per_episode = 1000):
        '''
        This class is our gym environment of a Box world with random objects placed inside
        self.action_space: Our robot's action space
        self.observation_space: Our robot's observation space (unsure whether pose or visuals)
        self.client: Our PyBullet physics client
        self.plane: The Box World env's plane floor
        self.spawn_height: The default height of each agent/object when spawned
        self.robots: Our four robots
        self.poses: Our four robots' initialization positions
        self.grab_distance: The distance that measures each robot's reach
        self.length: The length of each side of the box and also the length of the wall
        self.width: The width of only the wall
        self.objects: A list to keep track of the random objects we place
        self.num_objects: The number of random objects placed into the box
        self.num_objects: The initialized object positions
        self.img_w: Width of the camera
        self.fov: Scope of the camera
        self.distance: Idk what this is
        self.fpv_curr
        self.load_steps: The number of steps simulation takes to stepSimulation
        self.backpack
        self.force_scalar: The scalar multipled to the forces applied to the robots movement
        ## The remaining are object files
        '''
        self.is_save_images = is_save_images
        self.isGUI = isGUI
        self.Env_type = Env_type
        self.client = p.connect(p.GUI) if isGUI else p.connect(p.DIRECT)
        p.setTimeStep(1./240, self.client)
        self.plane = None
        self.spawn_height = 0.5
        self.num_robots = 1
        self.robot = None
        self.length = 12
        self.action_space = spaces.box.Box(
            low = np.array([0, 0], dtype=np.float32),
            high = np.array([self.length, self.length], dtype=np.float32))
        self.observation_space = spaces.box.Box(
            low = np.array([0], dtype=np.float32),
            high = np.array([255], dtype=np.float32))
        self.width = 0.1
        self.wall_height = 4
        self.objects = []
        self.num_objects = 3
        self.object_init_pos = []
        self.backpack = [(-1, ""), -1]
        self.num_correct_drops_for_obj = [0]*self.num_objects
        self.num_correct_drops = 0
        self.prev_picked_color = None
        self.img_w = 80
        self.obs_robot_init = np.zeros((self.img_w, self.img_w, self.img_w, 4), dtype=np.float32)
        self.fov = 90
        self.distance = 20
        self.fpv_prev = self.obs_robot_init
        self.fpv_curr = self.obs_robot_init
        self.bev = None
        self.max_episode_steps = max_steps_per_episode
        self.check_move_or_not = False
        self.load_steps = 100
        self.successful_picks = 0
        self.LOCAL_STEP_LIMIT = 30
        self.agent_name = os.path.join(os.path.dirname(__file__), './resources/agent.urdf')
        self.sphere_name = os.path.join(os.path.dirname(__file__), './resources/sphere2.urdf')
        self.cube_name = os.path.join(os.path.dirname(__file__), './resources/cube.urdf')
        self.cylinder_name = os.path.join(os.path.dirname(__file__), './resources/cylinder.urdf')
        self.cone_name = os.path.join(os.path.dirname(__file__), './resources/obj_files/cone_blue.obj')
        self.plane_name = os.path.join(os.path.dirname(__file__), './resources/plane.urdf')
    def reset(self):
        """
        Reset the environment, place robot in (6,6) and randomly initiale objects within box
        """
        p.resetSimulation(self.client)
        p.setGravity(0, 0, -9.8)
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-90, cameraTargetPosition=[self.length/2,self.length/2,0])
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.backpack = [(-1, ""), -1]
        self.check_stuck_list = [True] * 5 
        self.objects = []
        self.object_init_pos = []
        self.successful_picks = 0
        self.fpv_prev = self.obs_robot_init
        self.fpv_curr = self.obs_robot_init
        self.step_count = 0
        self.num_correct_drops = 0
        self.prev_picked_color = None
        self.bev = None
        self.one_hot = [0] * self.num_objects
        self.plane = p.loadURDF(fileName=self.plane_name, basePosition=[self.length/2,self.length/2,0])
        self.spawn_wall()
        self.spawn_robot()
        self.pre_pos = self.get_gtpose()
        self.curr_pos = self.get_gtpose()
        self.spawn_objects(init=True)
        self.load_camera()
        self.bev_target = self.bev
        self.spawn_objects() #rerandomize locations
         
        if self.Env_type == "Drop":  #### place the target object to the agent's backpack for drop task
            pos, ori = p.getBasePositionAndOrientation(self.objects[0][0])
            pos = [pos[0], pos[1], -10]
            p.resetBasePositionAndOrientation(self.objects[0][0], pos, ori)
            _, ori = p.getBasePositionAndOrientation(self.robot)
            pos = [pos[0], pos[1], self.spawn_height]
            p.resetBasePositionAndOrientation(self.robot, pos, ori)
            self.backpack = [self.objects[0], 0]
            
        self.load_camera()
        self.fpv_prev = self.fpv_curr # on first step, these should be same
        color_onehot = np.eye(6)[self.color_ind]
        type_onehot = np.eye(4)[self.objs_types_index]
        self.color_and_type_binary_vect = np.concatenate((color_onehot,type_onehot),axis=1)
        self.prev_score = 0
        self.rng = 1
        state = self.get_state(init=True)
        if self.Env_type == "Drop":
            self.prev_distance_object_to_target  = self.compute_distance_object_to_target()
        if self.Env_type == "Pick":
            self.prev_distance_agent_to_object = self.compute_distance_agent_to_object()
        if self.Env_type == "Decision":
            self.intial_energy = sum(self.compute_distance_object_to_target())
        if self.is_save_images:
            self.visualize(target=True) # for human gui, comment out for real
        return state

    def get_state(self, init=False):
        """
        Get current state
        input: init=T/F for creating state for target or any other step
        output: returns
        """
        raise NotImplementedError
    
    def visualize(self, target=False):
        """
        Save BEV, FPV(D), SPM, PCD images
        input: state = the current state; target = T/F whether target visualization or visualization after step
        """
        if target:
            target_bev = np.reshape(np.clip(self.bev_target, 0, 255), (160, 160, 4)).astype(np.uint8)
            plt.imshow(target_bev)
            filename = dir_path + "/imgs/target.png"
            plt.savefig(filename)
            plt.close()
            return
        bev = np.reshape(np.clip(self.bev, 0, 255), (160,160,4)).astype(np.uint8)
        target_bev = np.reshape(np.clip(self.bev_target, 0, 255), (160,160,4)).astype(np.uint8)
        fpv_curr = np.reshape(np.clip(self.fpv_curr, 0, 255), (self.img_w,self.img_w,3)).astype(np.uint8)
        f, axarr = plt.subplots(1,3) 
        axarr[0].imshow(target_bev)
        axarr[0].axis('off')
        axarr[1].imshow(bev)
        axarr[1].axis('off')
        axarr[2].imshow(fpv_curr)
        axarr[2].axis('off')
        filename = dir_path + "/imgs/timestep" + str(self.step_count) +".png"
        plt.savefig(filename)
        plt.close()
    
    def load_camera(self):
        """
        Take one frame of robot fpv and bev
        """
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        xB = xA + math.cos(yaw) * self.distance
        yB = yA + math.sin(yaw) * self.distance
        zB = zA
        view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[xA, yA, zA],
                            cameraTargetPosition=[xB, yB, zB],
                            cameraUpVector=[0, 0, 1.0])
        projection_matrix = p.computeProjectionMatrixFOV(
                                fov=self.fov, aspect=1.5, nearVal=0.02, farVal=self.distance)

        ## WE CAN ENABLE/DISABLE SHADOWS HERE
        robot_fpv = p.getCameraImage(self.img_w, self.img_w,
                                view_matrix,
                                projection_matrix, 
                                flags=p.ER_NO_SEGMENTATION_MASK)[2:4]
        bev_view_matrix = p.computeViewMatrix(
                            cameraEyePosition=[6, 6, 10],
                            cameraTargetPosition=[6, 5.99, 0.5],
                            cameraUpVector=[0,0,1])
        bev_projection_matrix = p.computeProjectionMatrixFOV(
                        fov=90, aspect=1, nearVal=0.02, farVal=100)
        bev = p.getCameraImage(BEV_PIXEL_WIDTH, BEV_PIXEL_WIDTH,
                            bev_view_matrix, bev_projection_matrix,
                            flags=p.ER_NO_SEGMENTATION_MASK)[2]
        seg_mask = set(list(p.getCameraImage(self.img_w, self.img_w,
                                view_matrix,
                                projection_matrix)[-1].flatten()))
        self.one_hot = [0] * self.num_objects
        for i in range(len(self.objects)):
            if self.objects[i][0] in seg_mask:
                self.one_hot[i] = 1
        fpv = cv2.cvtColor(np.array(robot_fpv[0], dtype=np.float32), cv2.COLOR_RGBA2RGB).flatten() # 80x80x4 (RGBA) might want to change this to RGB
        tdv = np.array(bev, dtype=np.float32).flatten() # 160x160x4
        self.fpv_prev = self.fpv_curr # (19200,)
        self.fpv_curr = fpv # (19200,)
        self.bev = tdv # (76800,)

    def spawn_robot(self):
        raise NotImplementedError
    
    def step(self, action, turn_step=10):
        raise NotImplementedError

    def move_agent(self, fwd_dist, fwd_drift):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        target = [pos[0]+math.cos(yaw)*fwd_dist, pos[1]+math.sin(yaw)*fwd_dist, pos[2]]
        self.collection_object = self.collision_detection(target)
        if self.collection_object > -1:
            return False
        p.resetBasePositionAndOrientation(self.robot, target, ori)
        return True

    def collision_detection(self, target):
        raise NotImplementedError

    def turn_agent(self, turn_angle):
        turn_angle *= math.pi / 180
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        new_ori = p.getEulerFromQuaternion(ori)
        new_ori = [new_ori[0],new_ori[1],new_ori[2]]
        new_ori[2] += turn_angle
        new_ori = p.getQuaternionFromEuler(new_ori)
        p.resetBasePositionAndOrientation(self.robot,pos,new_ori)

    def pick(self):
        """
        Robot grabs object in fpv and stores in virtual backpack
        """
        info = {"pick_status": "Pick failed bc nothing near"}
        grab_object = self.check_grab()
        if grab_object != -1:
            info = {"pick_status": "Picked properly"}
            obj = grab_object[0]
            idx = grab_object[1]
            pos, ori = p.getBasePositionAndOrientation(obj[0])
            pos = [pos[0], pos[1], -10]
            p.resetBasePositionAndOrientation(obj[0], pos, ori)  # hide the object
            self.backpack[0] = obj
            self.backpack[1] = idx
        return info

    def check_grab(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        grab_dist = 0.5 ** 0.5  # 1
        x_delta = math.cos(yaw) * grab_dist
        y_delta = math.sin(yaw) * grab_dist
        grab_pos = [pos[0] + x_delta, pos[1] + y_delta, 0]
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            dist = math.sqrt((pos[0] - grab_pos[0]) ** 2 + (pos[1] - grab_pos[1]) ** 2)
            grab_radius = 1  # 0.6
            if dist < grab_radius:
                return self.objects[i], i
        return -1

    def drop(self):
        if self.backpack[1] == -1:
            info = {"drop_status": "Failed drop bc backpack empty"}
            return info
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        x, y, _ = pos
        yaw = p.getEulerFromQuaternion(ori)[2]
        x_yaw = math.cos(yaw)
        y_yaw = math.sin(yaw)
        drop_dist = 1.5  # > 2*sqrt(0.5)
        x = x + x_yaw * drop_dist
        y = y + y_yaw * drop_dist
        success = self.collision_detection([x, y, 0])
        if success == -1:
            info = {"drop_status": "Dropped properly"}
            pos = [x, y, self.spawn_height]
            obj_id = self.backpack[0][0]
            if self.backpack[0][1] == "cone":
                ori = [math.pi / 2, 0, 0]
                ori = p.getQuaternionFromEuler(ori)
            p.resetBasePositionAndOrientation(obj_id, pos, ori)
            self.backpack = [(-1, ""), -1]
        else:
            info = {"drop_status": "Failed drop bc smth in the way"}
        return info

    def spawn_wall(self):
        """
        **FROM SPATIAL ACTION MAPS GITHUB**
        spawns the four surroundings walls
        """
        obstacle_color = (1, 1, 1, 1)
        obstacles = []
        for x, y, length, width in [
                (-self.width, 6, self.width, self.length+self.width),
                (self.length+self.width, 6, self.width, self.length+self.width),
                (6, -self.width, self.length+self.width, self.width),
                (6, self.length+self.width, self.length+self.width, self.width)
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'length': length, 'width': width})
        seen = []
        filename = os.path.join(os.path.dirname(__file__), './resources/wall_texture/wall_checkerboard_')
        for obstacle in obstacles:
            obstacle_half_extents = [obstacle['length'] / 2, obstacle['width'] / 2, self.wall_height]
            obstacle_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
            obstacle_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)
            obstacle_id = p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                [obstacle['position'][0], obstacle['position'][1], 0.5], p.getQuaternionFromEuler([0, 0, obstacle['heading']])
            )
            while True:
                id = random.randint(0,199)
                if id not in seen:
                    seen.append(id)
                    break
            x = p.loadTexture(filename + str(id) + ".png")
            p.changeVisualShape(obstacle_id, -1, textureUniqueId=x)
    def randomize_objs_pos(self):
        """
        Chooses random locations for the objects
        output: list size n for n objects with each item being (x,y)
        """
        robot_pos = p.getBasePositionAndOrientation(self.robot)[0]
        lastpos = [(robot_pos[0], robot_pos[1], self.spawn_height)]
        for i in range(self.num_objects):
            x = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
            y = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
            j = 0
            while j < i+self.num_robots:
                pos = lastpos[j]
                diff = math.sqrt((pos[0]-x)**2 +
                            (pos[1]-y)**2)
                if (diff <= 2*math.sqrt(0.5)):
                    x = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
                    y = random.uniform(math.sqrt(0.6),self.length-math.sqrt(0.6))
                    j = -1
                j += 1
            pos = [x,y,self.spawn_height]
            lastpos.append(pos)
        return lastpos[1:]

    def spawn_objects(self, init=False):
        """
        spawns the objects within the walls and no collision
        input: init = False; if True, store the init values
        """
        obj_poses = self.randomize_objs_pos()
        if init: 
            self.object_init_pos = obj_poses
            self.color_pool_name = ["green", "blue", "red", "skyblue", "yellow", "purple"]
            self.color_pool_value = [[0,1,0,1],[0, 0, 1, 1],[1, 0, 0, 1],[0,1,1,1], [1,1,0,1], [1,0,1,1]]
            self.color_ind = np.random.choice(6, self.num_objects, replace=False)
            self.objs_types_index = []
            for i in range(self.num_objects):
                choice = random.randint(0,3)
                self.objs_types_index.append(choice)
             
                pos = obj_poses[i]
                color_name_of_single_obj = self.color_pool_name[self.color_ind[i]]
                color_value_of_single_obj = self.color_pool_value[self.color_ind[i]]

                if choice == 0:
                    sphere_id = p.loadURDF(fileName=self.sphere_name, basePosition=pos)
                    p.changeVisualShape(sphere_id, -1, rgbaColor=color_value_of_single_obj)
                    self.objects.append((sphere_id, self.sphere_name))
                
                elif choice == 1:
                    yaw = random.uniform(0, 2*math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                    cube_id = p.loadURDF(fileName=self.cube_name, basePosition=pos, baseOrientation=ori)
                    p.changeVisualShape(cube_id, -1, rgbaColor=color_value_of_single_obj)
                    self.objects.append((cube_id , self.cube_name))
                    
                elif choice == 2:                
                    cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER)
                    cylinder_visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, rgbaColor=color_value_of_single_obj)
                    cylinder_id = p.createMultiBody(0, cylinder_collision_id, cylinder_visual_id, pos)
                    self.objects.append((cylinder_id, "cylinder"))
                    
                else:
                    ori = [math.pi/2,0,0]
                    ori = p.getQuaternionFromEuler(ori)
                    cone_collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.cone_name, meshScale=[0.5,0.5,0.5])
                    cone_visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.cone_name,
                                meshScale=[0.5,0.5,0.5], rgbaColor=color_value_of_single_obj)
                    cone_id = p.createMultiBody(0, cone_collision_id, cone_visual_id, pos, ori)
                    self.objects.append((cone_id,"cone"))

        else:
            for i in range(self.num_objects):
                _, ori = p.getBasePositionAndOrientation(self.objects[i][0])
                if self.objects[i][1] == self.cube_name: 
                    yaw = random.uniform(0, 2*math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                p.resetBasePositionAndOrientation(self.objects[i][0], obj_poses[i], ori)

    def backpack_onehot(self):
        onehot = [1,1,1]
        if self.backpack[1] != -1:
            onehot[self.backpack[1]] == 0
        return onehot

    def get_action_space(self):
        return BEV_PIXEL_WIDTH*BEV_PIXEL_WIDTH

    def seed(self, seed):
        """
        sets default seed based on rank
        input: seed = any integer
        """
        self.seed = seed
        random.seed(self.seed)

    def get_object_gtpose(self):
        poses = []
        for i in range(self.num_objects):
            if i == self.backpack[1]: 
                agent_pos = self.get_gtpose()[:2]
                poses.extend(agent_pos)
            else:
                gtpos = p.getBasePositionAndOrientation(self.objects[i][0])[0]
                poses.extend([gtpos[0], gtpos[1]])
        return poses
    def get_gtpose(self):
        """
        Return gt pose for lnet
        """
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        ori = p.getEulerFromQuaternion(ori)
        #return np.array([pos[0], pos[1], ori[2]])
        return [pos[0], pos[1], ori[2]]
    def close(self):
        """
        close the pybullet connection
        """
        p.disconnect(self.client)

class Environment_Decision(BoxWorld):
    def __init__(self, test=False, isGUI=False, is_save_images=False, max_steps_per_episode = 1000):
        super().__init__(Env_type="Decision", isGUI=isGUI, is_save_images=is_save_images, max_steps_per_episode=max_steps_per_episode)
        self.test = test
    def get_state(self, init=False):
        """
        Get current state
        input: init=T/F for creating state for target or any other step
        output: returns list of FPV, color vector, backpack, target position
        """
        state=[self.fpv_curr, self.color_and_type_binary_vect, np.array([self.backpack[1]]), np.array([self.object_init_pos[0][0:2], self.object_init_pos[1][0:2], self.object_init_pos[2][0:2]])]
        return np.array(state, dtype=object)

    def compute_distance_object_to_target(self):
        ds = []
        for i in range(self.num_objects):
            init_pos = self.object_init_pos[i]
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            d = ((pos[0] - init_pos[0]) ** 2 + (pos[1] - init_pos[1]) ** 2) ** 0.5
            ds.append(d)
        ds = np.array(ds)
        return ds
    
    def spawn_robot(self):
        pos = [6, 6, self.spawn_height]
        ori = p.getQuaternionFromEuler([0, 0, math.pi / 2])
        self.robot = p.loadURDF(fileName=self.agent_name, basePosition=pos, baseOrientation=ori)
    
    def step(self, action, turn_step=10):
        """
        Env takes one step, one timestep for robot movement and one frame for cameras
        input: action = integer able to fulfill one of these conditions
        output: next state
        """
        reward = -0.01
        done = False
        info = {}
        self.fwd_step, fwd_drift = np.random.normal(0.15, 0.01), np.random.normal(0, 0)
        self.turn_step = np.random.normal(turn_step, 1)
        if action == 0: # move forward
            self.check_move_or_not = self.move_agent(self.fwd_step, fwd_drift)
            self.check_stuck_list.append(self.check_move_or_not)
            self.check_stuck_list.pop(0)
        if action == 1: # turn left
            self.turn_agent(self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
        if action == 2: # turn right
            self.turn_agent(-self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
        elif action == 3:  # pick
            info = self.pick()
        elif action == 4:  # drop
            info = self.drop()
        for i in range(self.load_steps):
            p.stepSimulation()
        self.load_camera()
        self.step_count += 1
        self.curr_pos = self.get_gtpose()
        state = self.get_state()

        if (self.check_stuck_list == [False]*5) and (self.check_move_or_not == False):
            reward -= 1.0
            info.update({"move_status": "Stuck and cannot move"})
            done = True
        if self.is_save_images:
            self.visualize() # for human gui, comment out in real
        if action == 4 and self.backpack[1] == -1:
            ds = self.compute_distance_object_to_target()
            score = sum(self.compute_distance_object_to_target() < self.rng)
            if score > self.prev_score:
                info.update({"success_status": "Dropped an object into its initial position."})
            self.prev_score = score
            if score >= 3 - 0.5:
                reward+=10
                done = True
        if self.step_count >= self.max_episode_steps:
            done = True
            score = sum(self.compute_distance_object_to_target() < self.rng)
            info.update({"step_status": "reached max steps per episode"})
        if done:
            if self.test:
                final_energy = sum(self.compute_distance_object_to_target())
                Energy_Remaining = final_energy/self.intial_energy
                success_rate = self.prev_score / self.num_objects
                if success_rate == 1:
                    success = 1
                else:
                    success = 0
                info.update({"success": success,
                            "fixed_strict": success_rate,
                            "Energy_Remaining": Energy_Remaining
                            })
            else:
                success_rate = self.prev_score / self.num_objects
                info.update({"success_rate": success_rate})
        return state, reward, done, info
    
    def collision_detection(self, target):
        """
        Checks whether the target coordinate is not colliding with other objects/walls or is outside the map.
        input: target = (x,y)
        output: Returns -1 on wall collision, 1 on success, or the object item in self.objects
        """
        x, y, _ = target
        if x + math.sqrt(0.5) >= self.length or \
                x - math.sqrt(0.5) <= 0 or \
                y + math.sqrt(0.5) >= self.length or \
                y - math.sqrt(0.5) <= 0:
            return 3
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            diff = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            radius = math.sqrt(0.5)
            if self.objects[i][1] == self.cube_name:
                radius = math.sqrt(0.5)
            if diff < math.sqrt(0.5) + radius:
                return i
        return -1

class Environment_Pick(BoxWorld):
    def __init__(self, isGUI=False, is_save_images=False, max_steps_per_episode = 150):
        super().__init__(Env_type="Pick", isGUI=isGUI, is_save_images=is_save_images, max_steps_per_episode=max_steps_per_episode)
    
    def get_state(self, init=False):
        """
        Get current state
        input: init=T/F for creating state for target or any other step
        output: returns list of agent pos, one_hot for pick/drop, obj pcds, target spm, agent spm
        """
        state = [self.fpv_curr, self.color_and_type_binary_vect]
        return np.array(state, dtype=object)

    def compute_distance_agent_to_object(self):
        # the first object is the target object.
        agent_pos, _ = p.getBasePositionAndOrientation(self.robot)
        agent_pos = [agent_pos[0], agent_pos[1], 0]
        obj_pos, _ = p.getBasePositionAndOrientation(self.objects[0][0])
        obj_pos = [obj_pos[0], obj_pos[1], 0]
        dist = ((agent_pos[0] - obj_pos[0]) ** 2 + (agent_pos[1] - obj_pos[1]) ** 2) ** 0.5
        return dist

    def spawn_robot(self):
        # randomize agent's position
        x = random.uniform(math.sqrt(0.5), self.length - math.sqrt(0.5))
        y = random.uniform(math.sqrt(0.5), self.length - math.sqrt(0.5))
        pos = [x, y, self.spawn_height]
        ori = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])
        self.robot = p.loadURDF(fileName=self.agent_name, basePosition=pos, baseOrientation=ori)
    
    def move_agent(self, fwd_dist, fwd_drift):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        target = [pos[0] + math.cos(yaw) * fwd_dist, pos[1] + math.sin(yaw) * fwd_dist, pos[2]]
        self.collection_object = self.collision_detection(target)

        if self.collection_object != 1:
            return False

        p.resetBasePositionAndOrientation(self.robot, target, ori)
        return True

    def step(self, action, turn_step=10):
        """
        Env takes one step, one timestep for robot movement and one frame for cameras
        input: action = integer able to fulfill one of these conditions
        output: next state
        """
        reward = 0
        done = False
        info = {}
        self.fwd_step, fwd_drift = np.random.normal(0.15, 0.01), np.random.normal(0, 0)
        self.turn_step = np.random.normal(turn_step, 1)

        if action == 0:  # move forward
           
            self.check_move_or_not = self.move_agent(self.fwd_step, fwd_drift)
            self.check_stuck_list.append(self.check_move_or_not)
            self.check_stuck_list.pop(0)
       
        if action == 1:  # turn left
            self.turn_agent(self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
            
        if action == 2:  # turn right
            self.turn_agent(-self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
            
        if action == 3:  # pick
            self.pick()
            done = True

        for i in range(self.load_steps):
            p.stepSimulation()

        self.load_camera()
        self.step_count += 1
        self.curr_pos = self.get_gtpose()
        factor = 1.0
        curr_distance_agent_to_object = self.compute_distance_agent_to_object()
        reward_approaching = self.prev_distance_agent_to_object - curr_distance_agent_to_object
        reward += factor * reward_approaching
        state = self.get_state()
        if self.is_save_images:
            self.visualize() # for human gui, comment out in real
        info.update({"reward_approach": reward_approaching, "distance_to_object": curr_distance_agent_to_object})
        self.prev_distance_agent_to_object = curr_distance_agent_to_object
        if (self.check_stuck_list == [False] * 5) and (
                self.check_move_or_not == False and self.collection_object != self.objects[0]):
            reward -= 1.0
            info.update({"move_status": "Stuck and cannot move"})
            done = True
        if done:
            if self.backpack[1] == 0:  # if the object 0 has been picked up
                reward = 10
                info.update({"pick_status": "Successfully picked the target object."})
            else:
                reward -= 1 if self.step_count <= 50 else 0
                info.update({"pick_status": "Failed to pick the target object."})
        if self.step_count >= self.max_episode_steps:
            done = True
            info.update({"step_status": "reached max steps per episode"})
        return state, reward, done, info

    def collision_detection(self, target):
        """
        Checks whether the target coordinate is not colliding with other objects/walls or is outside the map.
        input: target = (x,y)
        output: Returns -1 on wall collision, 1 on success, or the object item in self.objects
        """
        x, y, _ = target
        if x + math.sqrt(0.6) >= self.length or \
                x - math.sqrt(0.6) <= 0 or \
                y + math.sqrt(0.6) >= self.length or \
                y - math.sqrt(0.6) <= 0:
            return -1
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            diff = math.sqrt((pos[0] - x) ** 2 +
                             (pos[1] - y) ** 2)
            radius = 0.5
            if self.objects[i][1] == self.cube_name:
                radius = math.sqrt(0.5)
            if diff < math.sqrt(0.5) + radius:
                return self.objects[i]
        return 1

class Environment_Drop(BoxWorld):
    def __init__(self, isGUI=False, is_save_images=False, max_steps_per_episode = 150):
        super().__init__(Env_type="Drop", isGUI=isGUI, is_save_images=is_save_images, max_steps_per_episode=max_steps_per_episode)

    def compute_distance_object_to_target(self):
        agent_pos, _ = p.getBasePositionAndOrientation(self.robot)
        x, y, _ = agent_pos
        target_pos = self.object_init_pos[0]
        dist = ((x - target_pos[0]) ** 2 + (y - target_pos[1]) ** 2) ** 0.5
        return dist

    def get_state(self, init=False):
        """
        Get current state
        input: init=T/F for creating state for target or any other step
        output: returns list of agent pos, one_hot for pick/drop, obj pcds, target spm, agent spm
        """
        state=[self.fpv_curr, self.color_and_type_binary_vect, np.array(self.object_init_pos[0][0:2])]
        return np.array(state, dtype=object)
    
    def spawn_robot(self):
        # randomize agent's position
        x = random.uniform(math.sqrt(0.5), self.length - math.sqrt(0.5))
        y = random.uniform(math.sqrt(0.5), self.length - math.sqrt(0.5))
        pos = [x, y, self.spawn_height]
        ori = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])
        self.robot = p.loadURDF(fileName=self.agent_name, basePosition=pos, baseOrientation=ori)
    
    def step(self, action, turn_step=10):
        """
        Env takes one step, one timestep for robot movement and one frame for cameras
        input: action = integer able to fulfill one of these conditions
        output: next state
        """
        reward = 0
        done = False
        info = {}
        self.fwd_step, fwd_drift = np.random.normal(0.15, 0.01), np.random.normal(0, 0)
        self.turn_step = np.random.normal(turn_step, 1)
        if action == 0: # move forward
            self.check_move_or_not = self.move_agent(self.fwd_step, fwd_drift)
            self.check_stuck_list.append(self.check_move_or_not)
            self.check_stuck_list.pop(0)
        if action == 1: # turn left
            self.turn_agent(self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
        if action == 2: # turn right
            self.turn_agent(-self.turn_step)
            self.check_stuck_list.append(True)
            self.check_stuck_list.pop(0)
        if action == 3: # drop
            self.drop()
            done = True
        for i in range(self.load_steps):
            p.stepSimulation()
        self.load_camera()
        self.step_count += 1
        self.curr_pos = self.get_gtpose()
        factor = 1.0
        curr_distance_object_to_target  = self.compute_distance_object_to_target()
        reward_approaching = self.prev_distance_object_to_target  - curr_distance_object_to_target 
        reward += factor * reward_approaching
        state = self.get_state()
        if self.is_save_images:
            self.visualize() # for human gui, comment out in real
        info.update({"reward_approach": reward_approaching, "distance_object_to_target": curr_distance_object_to_target})
        self.prev_distance_object_to_target  = curr_distance_object_to_target 
        if (self.check_stuck_list == [False]*5) and (self.check_move_or_not == False and self.collection_object != 0):
            reward -= 1.0
            info.update({"move_status": "Stuck and cannot move"})
            done = True
        if done:
            if self.backpack[1] == -1 and self.check_tidy():  # if the object reaches its initial position
                reward = 10
                info.update({"drop_status": "Successfully dropped the target object."})
            else:
                reward -= 1 if self.step_count <= 50 else 0
                info.update({"pick_status": "Failed to drop the target object."})
        if self.step_count >= self.max_episode_steps:
            done = True
            info.update({"step_status": "reached max steps per episode"})
        return state, reward, done, info

    def check_tidy(self):
        # check if the object 0 is in place
        pos = p.getBasePositionAndOrientation(self.objects[0][0])[0]
        pos = [pos[0], pos[1], 0]
        init_pos = [self.object_init_pos[0][0], self.object_init_pos[0][1], 0]
        dist = np.linalg.norm(np.array(pos) - np.array(init_pos))
        return dist < 1.0
    def collision_detection(self, target):
        """
        Checks whether the target coordinate is not colliding with other objects/walls or is outside the map.
        input: target = (x,y)
        output: Returns -1 on wall collision, 1 on success, or the object item in self.objects
        """
        x, y, _ = target
        if x + math.sqrt(0.5) >= self.length or \
                x - math.sqrt(0.5) <= 0 or \
                y + math.sqrt(0.5) >= self.length or \
                y - math.sqrt(0.5) <= 0:
            return 3
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            diff = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            radius = math.sqrt(0.5)
            if self.objects[i][1] == self.cube_name:
                radius = math.sqrt(0.5)
            if diff < math.sqrt(0.5) + radius:
                return i
        return -1