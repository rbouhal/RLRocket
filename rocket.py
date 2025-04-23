import numpy as np
import random
import cv2
import utils



class WindField:
    """
    Represents a wind field in the rocket environment.
    Can be either a constant velocity field in a specific area or a temporary gust.
    """
    def __init__(self, x_min, x_max, y_min, y_max, vx, vy, duration=None, start_time=None):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.vx = vx  # x-component of wind velocity
        self.vy = vy  # y-component of wind velocity
        self.duration = duration  # None means permanent wind field, otherwise temporary gust
        self.start_time = start_time  # When the gust starts (in simulation steps)
        
    def is_active(self, step_id):
        """Check if the wind field is active at the current step."""
        if self.duration is None:  # Permanent wind field
            return True
        elif self.start_time <= step_id < self.start_time + self.duration:
            return True
        return False
    
    def is_in_field(self, x, y):
        """Check if a point is inside the wind field."""
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max)
    
    def get_wind_velocity(self, x, y, step_id):
        """Get wind velocity at a given position and time."""
        if self.is_active(step_id) and self.is_in_field(x, y):
            return self.vx, self.vy
        return 0, 0


class Rocket(object):
    """
    Rocekt and environment.
    The rocket is simplified into a rigid body model with a thin rod,
    considering acceleration and angular acceleration and air resistance
    proportional to velocity.

    There are two tasks: hover and landing
    Their reward functions are straight forward and simple.

    For the hover tasks: the step-reward is given based on two factors
    1) the distance between the rocket and the predefined target point
    2) the angle of the rocket body (the rocket should stay as upright as possible)

    For the landing task: the step-reward is given based on three factors:
    1) the distance between the rocket and the predefined landing point.
    2) the angle of the rocket body (the rocket should stay as upright as possible)
    3) Speed and angle at the moment of contact with the ground, when the touching-speed
    are smaller than a safe threshold and the angle is close to 90 degrees (upright),
    we see it as a successful landing.

    """

    def __init__(self, max_steps, task='hover', rocket_type='falcon',
                 viewport_h=768, path_to_bg_img=None, enable_wind=True,
                 wind_difficulty=1):

        self.task = task
        self.rocket_type = rocket_type
        self.enable_wind = enable_wind
        self.wind_difficulty = wind_difficulty

        self.g = 9.8
        self.H = 50  # rocket height (meters)
        self.I = 1/12*self.H*self.H  # Moment of inertia
        self.dt = 0.05

        self.world_x_min = -300  # meters
        self.world_x_max = 300
        self.world_y_min = -30
        self.world_y_max = 570

        # target point
        if self.task == 'hover':
            self.target_x, self.target_y, self.target_r = 0, 200, 50
        elif self.task == 'landingai':
            self.target_x, self.target_y, self.target_r = 0, self.H/2.0, 50

        self.already_landing = False
        self.already_crash = False
        self.max_steps = max_steps

        # viewport height x width (pixels)
        self.viewport_h = int(viewport_h)
        self.viewport_w = int(viewport_h * (self.world_x_max-self.world_x_min) \
                          / (self.world_y_max - self.world_y_min))
        self.step_id = 0

        self.state = self.create_random_state()
        self.action_table = self.create_action_table()

        self.state_dims = 8
        self.action_dims = len(self.action_table)

        if path_to_bg_img is None:
            path_to_bg_img = task+'.jpg'
        self.bg_img = utils.load_bg_img(path_to_bg_img, w=self.viewport_w, h=self.viewport_h)

        self.state_buffer = []
    
        self.wind_fields = []
        self.scheduled_gusts = []
        if self.enable_wind:
            self._create_random_wind_fields()
            
    def _create_random_wind_fields(self):
        """Create random wind fields and scheduled gusts."""
        self.wind_fields = []
        self.scheduled_gusts = []
        
        # Create constant wind fields (jet streams)
        num_fields = random.randint(0, 2)
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        
        for _ in range(num_fields):
            # Determine position and size of wind field
            width = x_range #jet stream
            height = random.uniform(0.05 * y_range, 0.1 * y_range)

            #uncomment for non jet stream
            #width = random.uniform(0.05 * x_range, 0.15 * x_range)
            #center_x = random.uniform(self.world_x_min + width/2, self.world_x_max - width/2)

            # Place fields in the upper two-thirds of the environment
            center_x = self.world_x_min + x_range/2
            center_y = random.uniform(self.world_y_min, self.world_y_max)
            
            # Wind velocity (stronger with higher difficulty)
            max_wind = 20.0 * self.wind_difficulty
            vx = random.uniform(-max_wind, max_wind)*.1
            vy = random.uniform(-max_wind/2, max_wind/2)  # Vertical wind is usually less strong
            vy = 0
            field = WindField(
                x_min=center_x - width/2,
                x_max=center_x + width/2,
                y_min=center_y - height/2,
                y_max=center_y + height/2,
                vx=vx,
                vy=vy
            )
            
            self.wind_fields.append(field)
        
        # Schedule random gusts throughout the simulation
        num_gusts = random.randint(0, 2)
        for _ in range(num_gusts):
            # Determine when the gust will occur
            start_time = random.randint(1, int(self.max_steps * 0.8))
            duration = random.randint(5, 20)  # Duration in simulation steps
            
            # Determine position and size of gust
            #width = random.uniform(0.2 * x_range, 0.5 * x_range)
            width = x_range
            height = random.uniform(0.05 * y_range, 0.1 * y_range)
            
            #center_x = random.uniform(self.world_x_min + width/2, self.world_x_max - width/2)
            center_x = self.world_x_min + x_range/2
            center_y = random.uniform(self.world_y_min, self.world_y_max)
            
            # Gust velocity (stronger with higher difficulty)
            max_gust = 30.0 * self.wind_difficulty
            vx = random.uniform(-max_gust, max_gust)*.1
            vy = random.uniform(-max_gust/2, max_gust/2)
            vy = 0
            gust = WindField(
                x_min=center_x - width/2,
                x_max=center_x + width/2,
                y_min=center_y - height/2,
                y_max=center_y + height/2,
                vx=vx,
                vy=vy,
                duration=duration,
                start_time=start_time
            )
            
            self.scheduled_gusts.append(gust)

    def _get_wind_velocity(self, x, y):
        """Get the combined wind velocity at a given position."""
        vx_total, vy_total = 0, 0
        
        # Check constant wind fields
        for field in self.wind_fields:
            vx, vy = field.get_wind_velocity(x, y, self.step_id)
            vx_total += vx
            vy_total += vy
            
        # Check temporary gusts
        for gust in self.scheduled_gusts:
            vx, vy = gust.get_wind_velocity(x, y, self.step_id)
            vx_total += vx
            vy_total += vy
            
        return vx_total, vy_total

    def reset(self, state_dict=None):

        if state_dict is None:
            self.state = self.create_random_state()
        else:
            self.state = state_dict

        if self.enable_wind:
            self._create_random_wind_fields()

        self.state_buffer = []
        self.step_id = 0
        self.already_landing = False
        cv2.destroyAllWindows()
        return self.flatten(self.state)

    def create_action_table(self):
        f0 = 0
        f1 = 0.2 * self.g  # thrust
        f2 = 1.0 * self.g
        f3 = 2 * self.g
        vphi0 = 0  # Nozzle angular velocity
        vphi1 = 30 / 180 * np.pi
        vphi2 = -30 / 180 * np.pi

        action_table = [[f0, vphi0], [f0, vphi1], [f0, vphi2],
                        [f1, vphi0], [f1, vphi1], [f1, vphi2],
                        [f2, vphi0], [f2, vphi1], [f2, vphi2],
                        [f3, vphi0], [f3, vphi1], [f3, vphi2]
                        ]
        return action_table

    def get_random_action(self):
        return random.randint(0, len(self.action_table)-1)

    def create_random_state(self):

        # predefined locations
        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min
        xc = (self.world_x_max + self.world_x_min) / 2.0
        yc = (self.world_y_max + self.world_y_min) / 2.0


        if self.task == 'landingai':
            x = random.uniform(xc - x_range / 4.0, xc + x_range / 4.0)
            y = yc + 0.4*y_range
            if x <= 0:
                theta = -85 / 180 * np.pi
            else:
                theta = 85 / 180 * np.pi
            vy = -50

        if self.task == 'hover':
            x = xc
            y = yc + 0.2 * y_range
            theta = random.uniform(-45, 45) / 180 * np.pi
            vy = -10

        state = {
            'x': x, 'y': y, 'vx': 0, 'vy': vy,
            'theta': theta, 'vtheta': 0,
            'phi': 0, 'f': 0,
            't': 0, 'a_': 0,
            'wind_vx': 0, 'wind_vy': 0  # Track current wind velocity for display
        }

        return state

    def check_crash(self, state):
        if self.task == 'hover':
            x, y = state['x'], state['y']
            theta = state['theta']
            crash = False
            if y <= self.H / 2.0:
                crash = True
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            return crash

        elif self.task == 'landingai':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5

            crash = False
            if y >= self.world_y_max - self.H / 2.0:
                crash = True
            if y <= 0 + self.H / 2.0 and v >= 15.0:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(x) >= self.target_r:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(theta) >= 10/180*np.pi:
                crash = True
            if y <= 0 + self.H / 2.0 and abs(vtheta) >= 10/180*np.pi:
                crash = True
            return crash

    def check_landing_success(self, state):
        if self.task == 'hover':
            return False
        elif self.task == 'landingai':
            x, y = state['x'], state['y']
            vx, vy = state['vx'], state['vy']
            theta = state['theta']
            vtheta = state['vtheta']
            v = (vx**2 + vy**2)**0.5
            return True if y <= 0 + self.H / 2.0 and v < 15.0 and abs(x) < self.target_r \
                           and abs(theta) < 10/180*np.pi and abs(vtheta) < 10/180*np.pi else False

    def calculate_reward(self, state):

        x_range = self.world_x_max - self.world_x_min
        y_range = self.world_y_max - self.world_y_min

        # dist between agent and target point
        dist_x = abs(state['x'] - self.target_x)
        dist_y = abs(state['y'] - self.target_y)
        dist_norm = np.sqrt((dist_x / x_range)**2 + (dist_y / y_range)**2)

        dist_reward = 0.1*(1.0 - dist_norm)
        dist_reward = .2 * np.exp(-dist_norm)

        # Current angle and angular velocity
        theta = state['theta']
        vtheta = state['vtheta']
        moving_away = (theta * vtheta > 0)
        if abs(state['theta']) <= np.pi / 6.0:
            pose_reward = 0.1
        else:
            pose_reward = abs(state['theta']) / (0.5*np.pi)
            pose_reward = 0.1 * (1.0 - pose_reward)


        thrust = self.action_table[state['action_']][0]/ (np.max(np.array(self.action_table)[:,0]))
        fuel_penalty = -thrust * .05
        
        ang_v = abs(vtheta)
        if moving_away:
            ang_v_penalty = -ang_v * 0.4  # Higher penalty when moving away
        else:
            ang_v_penalty = +ang_v * 0.15  # Lower penalty when moving toward target
        

        reward = dist_reward + pose_reward + fuel_penalty +ang_v_penalty

        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 2*self.target_r:  # hit target
            reward = 0.25
        if self.task == 'hover' and (dist_x**2 + dist_y**2)**0.5 <= 1*self.target_r:  # hit target
            reward = 0.5
        if self.task == 'hover' and abs(state['theta']) > 90 / 180 * np.pi:
            reward = 0

        v = (state['vx'] ** 2 + state['vy'] ** 2) ** 0.5
        if self.task == 'landingai' and self.already_crash:
            reward = (reward + 5*np.exp(-1*v/10.)) * (self.max_steps - self.step_id)
        if self.task == 'landingai' and self.already_landing:
            reward = (1.0 + 5*np.exp(-1*v/10.))*(self.max_steps - self.step_id)

        return reward

    def step(self, action):
        x, y, vx, vy = self.state['x'], self.state['y'], self.state['vx'], self.state['vy']
        theta, vtheta = self.state['theta'], self.state['vtheta']
        phi = self.state['phi']
        f, vphi = self.action_table[action]

        ft, fr = -f*np.sin(phi), f*np.cos(phi)
        fx = ft*np.cos(theta) - fr*np.sin(theta)
        fy = ft*np.sin(theta) + fr*np.cos(theta)

        rho = 1 / (125/(self.g/2.0))**0.5  # suppose after 125 m free fall, then air resistance = mg
        ax, ay = fx-rho*vx, fy-self.g-rho*vy
        atheta = ft*self.H/2 / self.I

        # Apply wind effects if enabled
        wind_vx, wind_vy = 0, 0
        if self.enable_wind:
            wind_vx, wind_vy = self._get_wind_velocity(self.state['x'], self.state['y'])
            self.state['wind_vx'], self.state['wind_vy'] = wind_vx, wind_vy

        # Relative velocity including wind
        vx = vx + wind_vx
        vy = vy + wind_vy

        # update agent
        if self.already_landing:
            vx, vy, ax, ay, theta, vtheta, atheta = 0, 0, 0, 0, 0, 0, 0
            phi, f = 0, 0
            action = 0

        self.step_id += 1
        x_new = x + vx*self.dt + 0.5 * ax * (self.dt**2)
        y_new = y + vy*self.dt + 0.5 * ay * (self.dt**2)
        vx_new, vy_new = vx + ax * self.dt, vy + ay * self.dt
        theta_new = theta + vtheta*self.dt + 0.5 * atheta * (self.dt**2)
        vtheta_new = vtheta + atheta * self.dt
        phi = phi + self.dt*vphi

        phi = max(phi, -20/180*3.1415926)
        phi = min(phi, 20/180*3.1415926)

        self.state = {
            'x': x_new, 'y': y_new, 'vx': vx_new, 'vy': vy_new,
            'theta': theta_new, 'vtheta': vtheta_new,
            'phi': phi, 'f': f,
            't': self.step_id, 'action_': action,
            'wind_vx': wind_vx, 'wind_vy': wind_vy
        }
        self.state_buffer.append(self.state)

        self.already_landing = self.check_landing_success(self.state)
        self.already_crash = self.check_crash(self.state)
        reward = self.calculate_reward(self.state)

        if self.already_crash or self.already_landing:
            done = True
        else:
            done = False

        return self.flatten(self.state), reward, done, None

    def flatten(self, state):
        x = [state['x'], state['y'], state['vx'], state['vy'],
             state['theta'], state['vtheta'], state['t'],
             state['phi']]
        return np.array(x, dtype=np.float32)/100.

    def render(self, window_name='RLRocket', wait_time=1,
               with_trajectory=True, with_camera_tracking=True,
               crop_scale=0.4, show_wind = True):

        canvas = np.copy(self.bg_img)
        polys = self.create_polygons()
        #draw wind fields
        if show_wind and self.enable_wind:
            self.draw_wind_fields(canvas)
        # draw target region
        for poly in polys['target_region']:
            self.draw_a_polygon(canvas, poly)
        # draw rocket
        for poly in polys['rocket']:
            self.draw_a_polygon(canvas, poly)
        frame_0 = canvas.copy()

        # draw engine work
        for poly in polys['engine_work']:
            self.draw_a_polygon(canvas, poly)
        frame_1 = canvas.copy()

        if with_camera_tracking:
            frame_0 = self.crop_alongwith_camera(frame_0, crop_scale=crop_scale)
            frame_1 = self.crop_alongwith_camera(frame_1, crop_scale=crop_scale)

        # draw trajectory
        if with_trajectory:
            self.draw_trajectory(frame_0)
            self.draw_trajectory(frame_1)

        # draw text
        self.draw_text(frame_0, color=(0, 0, 0))
        self.draw_text(frame_1, color=(0, 0, 0))

        cv2.imshow(window_name, frame_0[:,:,::-1])
        cv2.waitKey(wait_time)
        cv2.imshow(window_name, frame_1[:,:,::-1])
        cv2.waitKey(wait_time)
        return frame_0, frame_1

    def draw_wind_fields(self, canvas):
        """Draw wind fields and gusts with visual indicators."""
        # Draw permanent wind fields
        for field in self.wind_fields:
            # Skip if not active
            if not field.is_active(self.step_id):
                continue
            
            # Convert world coordinates to pixels
            x1, y1 = self.wd2pxl([[field.x_min, field.y_min]])[0]
            x2, y2 = self.wd2pxl([[field.x_max, field.y_max]])[0]
            
            # Draw semi-transparent blue box for wind field
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (200, 100, 50), -1)  # Light blue
            alpha = 0.15  # Transparency
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
            
            # Draw arrows to indicate wind direction
            wind_magnitude = np.sqrt(field.vx**2 + field.vy**2)
            if wind_magnitude > 0:
                arrow_spacing = 4  # pixels
                arrow_length = min(30, max(10, int(wind_magnitude * 0.8)))  # Scale with wind strength
                
                # Draw a grid of arrows
                for ix in range(x1 + arrow_spacing, x2, arrow_spacing):
                    for iy in range(y1 + arrow_spacing, y2, arrow_spacing):
                        # Calculate arrow end point
                        angle = np.arctan2(field.vy, field.vx)
                        ex = int(ix + arrow_length * np.cos(angle))
                        ey = int(iy + arrow_length * np.sin(angle))
                        
                        # Draw arrow
                        cv2.arrowedLine(canvas, (ix, iy), (ex, ey), 
                                      (0, 0, 255), 1, tipLength=0.3)
        
        # Draw temporary gusts
        for gust in self.scheduled_gusts:
            # Skip if not active
            if not gust.is_active(self.step_id):
                continue
            

            # Convert world coordinates to pixels
            x1, y1 = self.wd2pxl([[gust.x_min, gust.y_min]])[0]
            x2, y2 = self.wd2pxl([[gust.x_max, gust.y_max]])[0]
            print(y1,'gust', y2)
            # Draw semi-transparent red box for gusts
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (50, 50, 220), -1)  # Light red
            
            # Calculate alpha based on remaining duration (fade out)
            remaining = gust.start_time + gust.duration - self.step_id
            alpha = 0.3 * (remaining / gust.duration)
            cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
            
            # Draw arrows to indicate wind direction
            wind_magnitude = np.sqrt(gust.vx**2 + gust.vy**2)
            if wind_magnitude > 0:
                arrow_spacing = 3  # pixels
                arrow_length = min(40, max(15, int(wind_magnitude)))  # Scale with wind strength
                
                # Draw a grid of arrows
                for ix in range(x1 + arrow_spacing, x2, arrow_spacing):
                    for iy in range(y1 + arrow_spacing, y2, arrow_spacing):
                        # Calculate arrow end point
                        angle = np.arctan2(gust.vy, gust.vx)
                        ex = int(ix + arrow_length * np.cos(angle))
                        ey = int(iy + arrow_length * np.sin(angle))
                        
                        # Draw arrow
                        cv2.arrowedLine(canvas, (ix, iy), (ex, ey), 
                                      (0, 0, 255), 2, tipLength=0.3)
    def create_polygons(self):

        polys = {'rocket': [], 'engine_work': [], 'target_region': []}

        if self.rocket_type == 'falcon':

            H, W = self.H, self.H/10
            dl = self.H / 30

            # rocket main body
            pts = [[-W/2, H/2], [W/2, H/2], [W/2, -H/2], [-W/2, -H/2]]
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})
            # rocket paint
            pts = utils.create_rectangle_poly(center=(0, -0.35*H), w=W, h=0.1*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            pts = utils.create_rectangle_poly(center=(0, -0.46*H), w=W, h=0.02*H)
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})
            # rocket landing rack
            pts = [[-W/2, -H/2], [-W/2-H/10, -H/2-H/20], [-W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})
            pts = [[W/2, -H/2], [W/2+H/10, -H/2-H/20], [W/2, -H/2+H/20]]
            polys['rocket'].append({'pts': pts, 'face_color': None, 'edge_color': (0, 0, 0)})

        elif self.rocket_type == 'starship':

            H, W = self.H, self.H / 2.6
            dl = self.H / 30

            # rocket main body (right half)
            pts = np.array([[ 0.        ,  0.5006878 ],
                           [ 0.03125   ,  0.49243465],
                           [ 0.0625    ,  0.48143053],
                           [ 0.11458334,  0.43878955],
                           [ 0.15277778,  0.3933975 ],
                           [ 0.2326389 ,  0.23796424],
                           [ 0.2326389 , -0.49931225],
                           [ 0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (242, 242, 242), 'edge_color': None})

            # rocket main body (left half)
            pts = np.array([[-0.        ,  0.5006878 ],
                           [-0.03125   ,  0.49243465],
                           [-0.0625    ,  0.48143053],
                           [-0.11458334,  0.43878955],
                           [-0.15277778,  0.3933975 ],
                           [-0.2326389 ,  0.23796424],
                           [-0.2326389 , -0.49931225],
                           [-0.        , -0.49931225]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (212, 212, 232), 'edge_color': None})

            # upper wing (right)
            pts = np.array([[0.15972222, 0.3933975 ],
                           [0.3784722 , 0.303989  ],
                           [0.3784722 , 0.2352132 ],
                           [0.22916667, 0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # upper wing (left)
            pts = np.array([[-0.15972222,  0.3933975 ],
                           [-0.3784722 ,  0.303989  ],
                           [-0.3784722 ,  0.2352132 ],
                           [-0.22916667,  0.23658872]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (42, 42, 42), 'edge_color': None})

            # lower wing (right)
            pts = np.array([[ 0.2326389 , -0.16368638],
                           [ 0.4548611 , -0.33562586],
                           [ 0.4548611 , -0.48555708],
                           [ 0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

            # lower wing (left)
            pts = np.array([[-0.2326389 , -0.16368638],
                           [-0.4548611 , -0.33562586],
                           [-0.4548611 , -0.48555708],
                           [-0.2638889 , -0.48555708]], dtype=np.float32)
            pts[:, 0] = pts[:, 0] * W
            pts[:, 1] = pts[:, 1] * H
            polys['rocket'].append({'pts': pts, 'face_color': (100, 100, 100), 'edge_color': None})

        else:
            raise NotImplementedError('rocket type [%s] is not found, please choose one '
                                      'from (falcon, starship)' % self.rocket_type)

        # engine work
        f, phi = self.state['f'], self.state['phi']
        c, s = np.cos(phi), np.sin(phi)

        if f > 0 and f < 0.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (237, 63, 28), 'edge_color': None})
        elif f > 0.5 * self.g and f < 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (237, 63, 28), 'edge_color': None})
        elif f > 1.5 * self.g:
            pts1 = utils.create_rectangle_poly(center=(2 * dl * s, -H / 2 - 2 * dl * c), w=dl, h=dl)
            pts2 = utils.create_rectangle_poly(center=(5 * dl * s, -H / 2 - 5 * dl * c), w=1.5 * dl, h=1.5 * dl)
            pts3 = utils.create_rectangle_poly(center=(8 * dl * s, -H / 2 - 8 * dl * c), w=2 * dl, h=2 * dl)
            pts4 = utils.create_rectangle_poly(center=(12 * dl * s, -H / 2 - 12 * dl * c), w=3 * dl, h=3 * dl)
            polys['engine_work'].append({'pts': pts1, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts2, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts3, 'face_color': (237, 63, 28), 'edge_color': None})
            polys['engine_work'].append({'pts': pts4, 'face_color': (237, 63, 28), 'edge_color': None})
        # target region
        if self.task == 'hover':
            pts1 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=0, h=self.target_r/3.0)
            pts2 = utils.create_rectangle_poly(center=(self.target_x, self.target_y), w=self.target_r/3.0, h=0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
        else:
            pts1 = utils.create_ellipse_poly(center=(0, 0), rx=self.target_r, ry=self.target_r/4.0)
            pts2 = utils.create_rectangle_poly(center=(0, 0), w=self.target_r/3.0, h=0)
            pts3 = utils.create_rectangle_poly(center=(0, 0), w=0, h=self.target_r/6.0)
            polys['target_region'].append({'pts': pts1, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts2, 'face_color': None, 'edge_color': (242, 242, 242)})
            polys['target_region'].append({'pts': pts3, 'face_color': None, 'edge_color': (242, 242, 242)})

        # apply transformation
        for poly in polys['rocket'] + polys['engine_work']:
            M = utils.create_pose_matrix(tx=self.state['x'], ty=self.state['y'], rz=self.state['theta'])
            pts = np.array(poly['pts'])
            pts = np.concatenate([pts, np.ones_like(pts)], axis=-1)  # attach z=1, w=1
            pts = np.matmul(M, pts.T).T
            poly['pts'] = pts[:, 0:2]

        return polys


    def draw_a_polygon(self, canvas, poly):

        pts, face_color, edge_color = poly['pts'], poly['face_color'], poly['edge_color']
        pts_px = self.wd2pxl(pts)
        if face_color is not None:
            cv2.fillPoly(canvas, [pts_px], color=face_color, lineType=cv2.LINE_AA)
        if edge_color is not None:
            cv2.polylines(canvas, [pts_px], isClosed=True, color=edge_color, thickness=1, lineType=cv2.LINE_AA)

        return canvas


    def wd2pxl(self, pts, to_int=True):

        pts_px = np.zeros_like(pts)

        scale = self.viewport_w / (self.world_x_max - self.world_x_min)
        for i in range(len(pts)):
            pt = pts[i]
            x_p = (pt[0] - self.world_x_min) * scale
            y_p = (pt[1] - self.world_y_min) * scale
            y_p = self.viewport_h - y_p
            pts_px[i] = [x_p, y_p]

        if to_int:
            return pts_px.astype(int)
        else:
            return pts_px

    def draw_text(self, canvas, color=(0, 0, 0)):

        def put_text(vis, text, pt):
            cv2.putText(vis, text=text, org=pt, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=color, thickness=1, lineType=cv2.LINE_AA)

        pt = (10, 20)
        text = "simulation time: %.2fs" % (self.step_id * self.dt)
        put_text(canvas, text, pt)

        pt = (10, 40)
        text = "simulation steps: %d" % (self.step_id)
        put_text(canvas, text, pt)

        pt = (10, 60)
        text = "x: %.2f m, y: %.2f m" % \
               (self.state['x'], self.state['y'])
        put_text(canvas, text, pt)

        pt = (10, 80)
        text = "vx: %.2f m/s, vy: %.2f m/s" % \
               (self.state['vx'], self.state['vy'])
        put_text(canvas, text, pt)

        pt = (10, 100)
        text = "a: %.2f degree, va: %.2f degree/s" % \
               (self.state['theta'] * 180 / np.pi, self.state['vtheta'] * 180 / np.pi)
        put_text(canvas, text, pt)


    def draw_trajectory(self, canvas, color=(0, 255, 0)):

        pannel_w, pannel_h = 256, 256
        traj_pannel = 255 * np.ones([pannel_h, pannel_w, 3], dtype=np.uint8)

        sw, sh = pannel_w/self.viewport_w, pannel_h/self.viewport_h  # scale factors

        # draw horizon line
        range_x, range_y = self.world_x_max - self.world_x_min, self.world_y_max - self.world_y_min
        pts = [[self.world_x_min + range_x/3, self.H/2], [self.world_x_max - range_x/3, self.H/2]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        # draw vertical line
        pts = [[0, self.H/2], [0, self.H/2+range_y/20]]
        pts_px = self.wd2pxl(pts)
        x1, y1 = int(pts_px[0][0]*sw), int(pts_px[0][1]*sh)
        x2, y2 = int(pts_px[1][0]*sw), int(pts_px[1][1]*sh)
        cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x2, y2),
                 color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        if len(self.state_buffer) < 2:
            return

        # draw traj
        pts = []
        for state in self.state_buffer:
            pts.append([state['x'], state['y']])
        pts_px = self.wd2pxl(pts)

        dn = 5
        for i in range(0, len(pts_px)-dn, dn):

            x1, y1 = int(pts_px[i][0]*sw), int(pts_px[i][1]*sh)
            x1_, y1_ = int(pts_px[i+dn][0]*sw), int(pts_px[i+dn][1]*sh)

            cv2.line(traj_pannel, pt1=(x1, y1), pt2=(x1_, y1_), color=color, thickness=2, lineType=cv2.LINE_AA)

        roi_x1, roi_x2 = self.viewport_w - 10 - pannel_w, self.viewport_w - 10
        roi_y1, roi_y2 = 10, 10 + pannel_h
        canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] = 0.6*canvas[roi_y1:roi_y2, roi_x1:roi_x2, :] + 0.4*traj_pannel



    def crop_alongwith_camera(self, vis, crop_scale=0.4):
        x, y = self.state['x'], self.state['y']
        xp, yp = self.wd2pxl([[x, y]])[0]
        crop_w_half, crop_h_half = int(self.viewport_w*crop_scale), int(self.viewport_h*crop_scale)
        # check boundary
        if xp <= crop_w_half + 1:
            xp = crop_w_half + 1
        if xp >= self.viewport_w - crop_w_half - 1:
            xp = self.viewport_w - crop_w_half - 1
        if yp <= crop_h_half + 1:
            yp = crop_h_half + 1
        if yp >= self.viewport_h - crop_h_half - 1:
            yp = self.viewport_h - crop_h_half - 1

        x1, x2, y1, y2 = xp-crop_w_half, xp+crop_w_half, yp-crop_h_half, yp+crop_h_half
        vis = vis[y1:y2, x1:x2, :]

        vis = cv2.resize(vis, (self.viewport_w, self.viewport_h))
        return vis
