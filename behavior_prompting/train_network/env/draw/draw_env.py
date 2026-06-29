from typing import Optional
import gym
from gym import spaces
import math
import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from behavior_prompting.train_network.env.draw.pymunk_override import DrawOptions
from behavior_prompting.train_network.utils.draw_util import compute_draw_distance

# Boundary angle bounds used in _setup() when randomize_boundary_angle is True
BOUNDARY_ANGLE_LOW = -np.pi / 4
BOUNDARY_ANGLE_HIGH = np.pi / 4


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class DrawEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            damping=None,
            render_size=224,
            render_cache_size=None,
            reset_to_state=None,
            boundary_angle=None,
            render_mode='rgb_array',
            target_drawing=None,
            target_boundary_angle=None,
            overlay_reward=True,
            overlay_action_cross=True,
            overlay_target_drawing=True,
            clock_hz=-1, # -1 if you want to disable the clock, otherwise the maximum frequency the environment will run at (useful for rolling out a policy that is visually displayed to the user)
            put_reward_in_title=False,
            enable_keyboard_control=False
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.render_cache_size = render_cache_size if render_cache_size is not None else render_size
        assert self.render_cache_size <= self.window_size, f"render_cache_size ({self.render_cache_size}) must be less than or equal to window_size ({self.window_size})"
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.randomize_boundary_angle = boundary_angle is None
        self.board_length = 350

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            ),
            'pen_down': spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float32
            )
        })

        # possible agent positions + pen up/down state
        self.action_space = spaces.Box(
            low=np.array([0,0,0], dtype=np.float64),
            high=np.array([ws,ws,1], dtype=np.float64),
            shape=(3,),
            dtype=np.float64
        )
        self.damping = damping
        self.render_cache = None
        
        # Persistent drawing canvas for efficient rendering
        self.drawing_canvas = None
        self.last_drawn_position = None
        self._saved_canvas_state = None

        self.overlay_reward = overlay_reward
        self.overlay_action_cross = overlay_action_cross
        self.overlay_target_drawing = overlay_target_drawing
        self.put_reward_in_title = put_reward_in_title
        self.enable_keyboard_control = enable_keyboard_control

        # Boundary configuration
        self.boundary_angle = boundary_angle  # If None, will be randomized in _setup()

        self.mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.clock_hz = clock_hz
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state

        self.set_target_drawing(target_drawing, target_boundary_angle)
    
    def reset(self, no_rotation: bool=False):
        # Check if we have a saved canvas state to restore
        if self._saved_canvas_state is not None:
            self.restore_canvas_state()
            self._saved_canvas_state = None # only restore once
        else:
            # Normal reset: set agent state
            seed = self._seed
            self._setup(no_rotation=no_rotation)
            if self.damping is not None:
                self.space.damping = self.damping
            
            state = self.reset_to_state
            if state is None:
                rs = np.random.RandomState(seed=seed)
                state = np.array([
                    rs.randint(50, 450), rs.randint(50, 450),
                    0 # pen up
                    ])
            self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            
            # Set pen state
            pen_is_down = action[2] > 0.5
            self.agent.pen_down = pen_is_down
            
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action[:2] - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)
            
            # Record agent position for drawing trail
            if self.drawing_canvas is not None:
                current_pos = np.array(self.agent.position)
                pygame_pos = pymunk.pygame_util.to_pygame(current_pos, self.drawing_canvas)
                
                if pen_is_down:
                    # Draw incrementally - only draw line from last position to current
                    if self.last_drawn_position is not None:
                        pygame.draw.line(self.drawing_canvas, (0, 0, 255), 
                                       self.last_drawn_position, pygame_pos, 12)
                    # Draw a small circle at current position for smoothness
                    pygame.draw.circle(self.drawing_canvas, (0, 0, 255), pygame_pos, 4)
                    self.last_drawn_position = pygame_pos
                else:
                    # Pen is up - reset last position
                    self.last_drawn_position = None

        reward = self.compute_reward()
        done = False
        observation = self._get_obs()
        info = self._get_info()

        if self.enable_keyboard_control:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        done = True

        return observation, reward, done, info
    
    def compute_reward(self) -> float:
        # Convert drawing canvas to numpy array
        canvas_array = pygame.surfarray.array3d(self.drawing_canvas)
        canvas_array = np.transpose(canvas_array, (1, 0, 2))  # Transpose to match numpy convention

        # Create a rotated version of the canvas
        canvas_center = (canvas_array.shape[0] // 2, canvas_array.shape[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(canvas_center, np.degrees(self.boundary_angle), 1.0)
        rotated_canvas = cv2.warpAffine(canvas_array, rotation_matrix, canvas_array.shape[:2], borderValue=(255, 255, 255))

        # Convert to binary masks (1 where drawn, 0 elsewhere)
        canvas_mask = rotated_canvas[:,:,0] == 0  # red channel will have 0 in drawn pixel locations since pen marker is pure blue
        target_mask = self.target_drawing[:,:,0] == 0 if self.target_drawing is not None else np.zeros_like(canvas_mask, dtype=bool)

        reward = -compute_draw_distance(target_mask, canvas_mask, center=False)

        if self.put_reward_in_title:
            pygame.display.set_caption(f"Reward: {reward:.2f}")
        
        return reward

    def render(self, mode):
        assert mode == self.mode
        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
                left_mouse_button = (np.float32(pygame.mouse.get_pressed()[0]),)
                act = act + left_mouse_button
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        img, canvas_with_target_img = self._render_frame()

        agent_pos = np.array(self.agent.position)
        pen_down = np.array([self.agent.pen_down])
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos,
            'pen_down': pen_down
        }

        # draw action
        if self.latest_action is not None and self.overlay_action_cross:
            self._draw_action_cross(canvas_with_target_img, self.latest_action)

        # optionally draw reward text
        if self.overlay_reward:
            reward = self.compute_reward()
            self._draw_reward(canvas_with_target_img, reward)

        self.render_cache = canvas_with_target_img

        return obs
    
    def _get_info(self):
        return {}

    def _render_frame(self):
        if self.window is None and self.mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.mode == "human":
            self.clock = pygame.time.Clock()

        target_canvases = []
        obs_canvas = pygame.Surface((self.window_size, self.window_size))
        obs_canvas.fill((255, 255, 255))
        target_canvases.append(obs_canvas)
        self.screen = obs_canvas

        canvas_with_target = pygame.Surface((self.window_size, self.window_size))
        canvas_with_target.fill((255, 255, 255))
        target_canvases.append(canvas_with_target)

        # Draw the pymunk objects (boundary, helper lines) on the background
        for canvas in target_canvases:
            draw_options = DrawOptions(canvas)
            self.space.debug_draw(draw_options)
        
        # Draw the target on the canvas_with_target
        self._draw_target(canvas_with_target)

        # Draw the persistent drawing trail on top of the boundary
        if self.drawing_canvas is not None:
            # Blit the persistent drawing canvas onto the main canvas
            # The drawing canvas has transparent white, so only the drawn lines show
            for canvas in target_canvases:
                canvas.blit(self.drawing_canvas, (0, 0))

        # Draw the agent on top of everything (ensure it's visible)
        if hasattr(self, 'agent') and self.agent is not None:
            # Draw agent as a green circle
            for canvas in target_canvases:
                agent_pygame_pos = pymunk.pygame_util.to_pygame(self.agent.position, canvas)    
                pygame.draw.circle(canvas, (48, 156, 54), agent_pygame_pos, 15)
                # Add a white outline to make it more visible
                pygame.draw.circle(canvas, (255, 255, 255), agent_pygame_pos, 15, 2)

        if self.mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas_with_target, canvas_with_target.get_rect())
            pygame.event.pump()
            pygame.display.update()

            if self.clock_hz != -1:
                self.clock.tick(self.clock_hz)

        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(obs_canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))

        canvas_with_target_img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas_with_target)), axes=(1, 0, 2)
            )
        canvas_with_target_img = cv2.resize(canvas_with_target_img, (self.render_cache_size, self.render_cache_size))
        
        return img, canvas_with_target_img
    
    def get_drawing_image(self):
        tmp_canvas = pygame.Surface((self.window_size, self.window_size))
        tmp_canvas.fill((255, 255, 255))
        tmp_canvas.blit(self.drawing_canvas, (0, 0))
        drawing_img = np.transpose(
                np.array(pygame.surfarray.pixels3d(tmp_canvas)), axes=(1, 0, 2)
            )
        return drawing_img

    def save_canvas_state(self):
        """Stores a copy of the current drawing canvas for later restoration."""
        assert self.drawing_canvas is not None, "Drawing canvas is not initialized"
        self._saved_canvas_state = self.drawing_canvas.copy()

    def restore_canvas_state(self):
        """Restores the drawing canvas from a previously saved copy."""
        assert self._saved_canvas_state is not None, "No canvas state to restore"
        self.drawing_canvas = self._saved_canvas_state.copy()
        self.drawing_canvas.set_colorkey((255, 255, 255))
        self.last_drawn_position = None  # avoid connecting new strokes to old ones
        # Put the pen up (keep position the same)
        if hasattr(self, 'agent') and self.agent is not None:
            self.agent.pen_down = False
    
    def _draw_target(self, canvas):
        if self.target_drawing is None or not self.overlay_target_drawing:
            return
        
        target_drawing = self.target_drawing[:,:,::-1] # flip the color channels so that it's not the same color as the new character that's going to be drawn

        # Transpose width and height dimensions (pygame has a different convention)
        target_drawing = np.transpose(target_drawing, (1, 0, 2))
        
        # Convert target drawing to pygame surface
        target_surface = pygame.surfarray.make_surface(target_drawing)
        target_surface.set_colorkey((255, 255, 255))  # Make white transparent
        
        # Rotate the surface
        rotated_surface = pygame.transform.rotate(target_surface, -np.degrees(self.boundary_angle))
        
        # Get rotated dimensions
        rotated_width, rotated_height = rotated_surface.get_size()
        
        # Calculate centering position
        # Center the rotated surface on the canvas
        x_offset = (self.window_size - rotated_width) // 2
        y_offset = (self.window_size - rotated_height) // 2
        
        # Blit the rotated target onto the canvas at the centered position
        canvas.blit(rotated_surface, (x_offset, y_offset))

    def _draw_action_cross(self, img, action):
        action = np.array(action)
        coord = (action[:2] / self.window_size * self.render_cache_size).astype(np.int32)  # Only use x, y position
        marker_size = int(8/96*self.render_cache_size)
        thickness = int(1/96*self.render_cache_size)
        cv2.drawMarker(img, coord,
            color=(255,0,0), markerType=cv2.MARKER_CROSS,
            markerSize=marker_size, thickness=thickness)

    def _draw_reward(self, img, reward):
        reward_text = f"Reward: {reward:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5 * (self.render_cache_size / 224.0)  # Scale font based on render_cache_size (224 is the default render_size)
        thickness = max(1, int(1 * (self.render_cache_size / 224.0)))  # Scale thickness based on render_cache_size
        color = (0, 0, 0)  # Black text
        position = (int(10 * (self.render_cache_size / 224.0)), int(20 * (self.render_cache_size / 224.0)))  # Scale position based on render_cache_size
        
        cv2.putText(img, reward_text, position, font, font_scale, color, thickness)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pen_down = state[2]
        self.agent.position = pos_agent
        self.agent.pen_down = pen_down
        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)

    def _create_rotated_square_boundary(self, angle=0):
        """Create a rotated square boundary with the given parameters."""        
        # Half side length for easier calculation
        half_side = self.board_length / 2
        
        # Define square corners relative to center (before rotation)
        corners = [
            (-half_side, -half_side),  # Bottom-left
            (half_side, -half_side),   # Bottom-right  
            (half_side, half_side),    # Top-right
            (-half_side, half_side)    # Top-left
        ]
        
        # Rotate corners around center
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        center_x = self.window_size / 2
        center_y = self.window_size / 2

        rotated_corners = []
        for x, y in corners:
            # Rotate point around origin
            rotated_x = x * cos_a - y * sin_a
            rotated_y = x * sin_a + y * cos_a
            # Translate to center position
            rotated_corners.append((rotated_x + center_x, rotated_y + center_y))
        
        # Store corner positions for helper lines
        # Order: bottom-left, bottom-right, top-right, top-left
        self.boundary_corners = rotated_corners
        
        # Calculate midpoints for helper lines
        bl, br, tr, tl = rotated_corners
        self.boundary_midpoints = {
            'top': ((tl[0] + tr[0])/2, (tl[1] + tr[1])/2),      # Top midpoint
            'bottom': ((bl[0] + br[0])/2, (bl[1] + br[1])/2),   # Bottom midpoint  
            'left': ((tl[0] + bl[0])/2, (tl[1] + bl[1])/2),     # Left midpoint
            'right': ((tr[0] + br[0])/2, (tr[1] + br[1])/2)     # Right midpoint
        }
        
        # Create wall segments connecting the corners
        walls = []
        for i in range(4):
            start = rotated_corners[i]
            end = rotated_corners[(i + 1) % 4]  # Connect to next corner (wrapping to 0)
            color = (255,0,0) if i == 0 else (0,0,0)
            wall = self._add_segment(start, end, 4, color)
            walls.append(wall)
        
        return walls

    def _create_helper_lines(self, color='LightGray', width=4):
        """Create helper lines for the boundary."""
        helpers = []
        if self.boundary_corners is not None:
            bl, br, tr, tl = self.boundary_corners  # bottom-left, bottom-right, top-right, top-left
            helpers.append(self._add_segment(tl, br, width, color))
            helpers.append(self._add_segment(tr, bl, width, color))
        if self.boundary_midpoints is not None:
            mid = self.boundary_midpoints
            helpers.append(self._add_segment(mid['top'], mid['bottom'], width, color))
            helpers.append(self._add_segment(mid['left'], mid['right'], width, color))
        return helpers
            
    def _setup(self, no_rotation: bool=False):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Initialize boundary geometry storage
        self.boundary_corners = []
        self.boundary_midpoints = {}
        
        # Set boundary angle if not specified
        if self.randomize_boundary_angle:
            # Random angle between BOUNDARY_ANGLE_LOW and BOUNDARY_ANGLE_HIGH
            self.boundary_angle = self.np_random.uniform(BOUNDARY_ANGLE_LOW, BOUNDARY_ANGLE_HIGH)
        
        if no_rotation:
            self.boundary_angle = 0.0
        
        # Add rotated square walls and helper lines
        walls = self._create_rotated_square_boundary(angle=self.boundary_angle)
        helpers = self._create_helper_lines()
        self.space.add(*helpers)
        self.space.add(*walls)
        
        # Add agent, block, and goal zone.
        center_x = self.window_size / 2
        center_y = self.window_size / 2
        self.agent = self.add_circle((center_x, center_y), 15)
        
        # Initialize/clear persistent drawing canvas
        self.drawing_canvas = pygame.Surface((self.window_size, self.window_size))
        self.drawing_canvas.fill((255, 255, 255))  # White background
        self.drawing_canvas.set_colorkey((255, 255, 255))  # Make white transparent
        self.last_drawn_position = None
        self._saved_canvas_state = None

    def _add_segment(self, a, b, radius, color='LightGray'):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color(color)    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        body.pen_down = False
        return body

    def set_target_drawing(self, target_drawing: Optional[np.ndarray], target_boundary_angle: Optional[float]=None):
        """Sets the target drawing to be displaying only during data collection as a reference for the human data collector. If target_drawing is None, the target drawing is not displayed. We assume the target drawing has no rotation and is in uint8 format with shape matching the window size: (512, 512, 3)."""
        if target_drawing is None:
            self.target_drawing = None
            return

        assert target_boundary_angle is not None

        # the target drawing is currently rotated by boundary angle. Undo this rotation to make the target drawing upright
        # Create a rotated version of the canvas
        canvas_center = (target_drawing.shape[0] // 2, target_drawing.shape[1] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(canvas_center, np.degrees(target_boundary_angle), 1.0)
        target_drawing = cv2.warpAffine(target_drawing, rotation_matrix, target_drawing.shape[:2], borderValue=(255, 255, 255))

        # the target drawing may have been JPEG compressed, so we need to handle any compression artifacts
        # find the pixel locations of target_drawing that are red
        tolerance = 50
        blue_mask = (target_drawing[:,:,0] <= tolerance) & (target_drawing[:,:,1] <= tolerance) & (target_drawing[:,:,2] >= 255 - tolerance)

        # convert non-blue pixels to white
        target_drawing[~blue_mask] = [255,255,255]

        # convert blue pixels to full blue
        target_drawing[blue_mask] = [0,0,255]

        self.target_drawing = target_drawing
