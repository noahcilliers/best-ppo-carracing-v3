"""MultiCarRacing — N cars racing on one shared CarRacing track.

This is a fork of gymnasium's ``CarRacing-v3`` (``gymnasium/envs/box2d/car_racing.py``)
with the multi-agent modifications from the old ``gym``-0.17 ``MultiCarRacing``
repo re-applied on top of the *current* pygame-based environment. We fork the new
env rather than patch the old one because the old one rendered through a pyglet /
OpenGL stack that no longer exists in gymnasium; the new env already solves
rendering, and its per-call fresh-surface design makes N independent per-car
cameras straightforward.

What changed relative to single-car CarRacing:
  * N cars share one Box2D world, spawned non-overlapping on the start line.
  * Per-car competitive tile reward (first car to a tile gets the most).
  * The contact listener routes each tile visit to the wheel's owning car.
  * step() takes a ``(N, 3)`` action and returns ``(N, 96, 96, 3)`` obs and a
    ``(N,)`` reward; per-car termination is exposed in ``info``.
  * Each car renders its own egocentric 96x96 view (all cars visible in each).
  * Optional car-to-car Box2D collisions (off by default; see ``collisions``).

When ``num_agents == 1`` the env reduces *exactly* to single-car CarRacing:
obs is ``(96, 96, 3)``, reward is a float, ``terminated`` is a bool — so it
passes SB3's ``check_env`` and hosts the existing policy unchanged.

The three physics-independent pieces — observation preprocessing, spawn geometry,
and the reward formula — live in sibling modules (``preprocessing``, ``spawn``,
``rewards``) so they can be reasoned about and tested away from Box2D/pygame.
"""

import math

import numpy as np

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled, InvalidAction
from gymnasium.utils import EzPickle

from multicar.rewards import competitive_tile_reward
from multicar.spawn import compute_spawn_poses

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, run `pip install swig` then `pip install "gymnasium[box2d]"`'
    ) from e

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e


# --- constants (copied from gymnasium CarRacing so we don't couple to its module) ---
STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0
TRACK_RAD = 900 / SCALE
PLAYFIELD = 2000 / SCALE
FPS = 50
ZOOM = 2.7

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE

# Per-car hull colors (0..1 floats — Car.draw multiplies by 255 itself).
CAR_COLORS = [
    (0.8, 0.0, 0.0),  # red   (the CarRacing default)
    (0.0, 0.0, 0.8),  # blue
    (0.0, 0.8, 0.0),  # green
    (0.8, 0.8, 0.0),  # yellow
    (0.8, 0.0, 0.8),  # magenta
    (0.0, 0.8, 0.8),  # cyan
    (0.9, 0.5, 0.0),  # orange
    (0.5, 0.0, 0.9),  # purple
]


class MultiFrictionDetector(contactListener):
    """Per-car version of CarRacing's FrictionDetector.

    The single-car listener marks each tile visited once and adds a flat bonus.
    Here every tile tracks visits *per car* (``tile.road_visited`` is a list),
    each wheel is tagged with its owning ``car_id``, and the reward routed to
    that car is damped by how many cars already reached the tile.
    """

    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        tile.color[:] = self.env.road_color
        # ``obj`` matters only if it is a wheel (wheels carry a ``tiles`` set).
        if not obj or "tiles" not in obj.__dict__:
            return

        if begin:
            obj.tiles.add(tile)
            car_id = obj.car_id
            if not tile.road_visited[car_id]:
                tile.road_visited[car_id] = True
                # how many OTHER cars reached this tile before this one
                past_visitors = sum(tile.road_visited) - 1
                self.env.tile_visited_count[car_id] += 1
                self.env.reward[car_id] += competitive_tile_reward(
                    self.env.num_agents, len(self.env.track), past_visitors
                )
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count[car_id] / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap[car_id] = True
        else:
            obj.tiles.remove(tile)


class MultiCarRacing(gym.Env, EzPickle):
    """N-car CarRacing. See module docstring for the full contract.

    Args:
        num_agents: number of cars sharing the track (default 2).
        render_mode: ``None``, ``"human"`` (tiled window), ``"rgb_array"``, or
            ``"state_pixels"``.
        verbose: print track-generation diagnostics.
        lap_complete_percent: fraction of tiles a car must visit for a lap.
        domain_randomize: randomize background/track colors each reset.
        continuous: continuous ``[steer, gas, brake]`` (True) or discrete (False).
        random_spawn: shuffle the car->grid-slot assignment each reset so no car
            has a fixed positional advantage.
        collisions: if True, cars physically collide with one another (hull-hull).
            Off by default so a race can first be validated without contact.
        max_episode_steps: internal truncation limit, so the env is self-contained
            even when constructed directly (not via ``gym.make`` + ``TimeLimit``).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "state_pixels"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        num_agents: int = 2,
        render_mode: str | None = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = True,
        random_spawn: bool = True,
        collisions: bool = False,
        max_episode_steps: int | None = 1000,
    ):
        EzPickle.__init__(
            self,
            num_agents,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
            random_spawn,
            collisions,
            max_episode_steps,
        )
        assert num_agents >= 1, "num_agents must be >= 1"
        self.num_agents = num_agents
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        self.lap_complete_percent = lap_complete_percent
        self.random_spawn = random_spawn
        self.collisions = collisions
        self.max_episode_steps = max_episode_steps
        self.verbose = verbose
        self._init_colors()

        self.contactListener_keepref = MultiFrictionDetector(self, lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: pygame.Surface | None = None
        self.surf = None
        self.clock = None
        self.isopen = True
        self.road = None
        self.cars: list[Car] = []
        # per-car reward bookkeeping (set fully in reset)
        self.reward = np.zeros(num_agents)
        self.prev_reward = np.zeros(num_agents)
        self.new_lap = [False] * num_agents
        self.fd_tile = fixtureDef(
            shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)])
        )

        # Action / observation spaces describe a SINGLE car — each per-car policy
        # sees single-car shapes. With num_agents > 1 the env returns stacked
        # (N, ...) arrays; with num_agents == 1 it returns exactly these shapes.
        if self.continuous:
            self.action_space = spaces.Box(
                np.array([-1, 0, 0]).astype(np.float32),
                np.array([+1, +1, +1]).astype(np.float32),
            )
        else:
            self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Box(
            low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8
        )

        self.render_mode = render_mode
        self._grid = None
        self._tile_size = None

    # ------------------------------------------------------------------ colors
    def _init_colors(self):
        if self.domain_randomize:
            self.road_color = self.np_random.uniform(0, 210, size=3)
            self.bg_color = self.np_random.uniform(0, 210, size=3)
            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert self.domain_randomize, "domain_randomize must be True to use this."
        if randomize:
            self.road_color = self.np_random.uniform(0, 210, size=3)
            self.bg_color = self.np_random.uniform(0, 210, size=3)
            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    # ------------------------------------------------------------------ destroy
    def _destroy(self):
        if self.road:
            for t in self.road:
                self.world.DestroyBody(t)
            self.road = []
        for car in self.cars:
            car.destroy()
        self.cars = []

    # -------------------------------------------------------------- collisions
    def _set_car_collisions(self, car, enabled):
        """Toggle whether this car collides with the OTHER cars.

        Box2D rule: two fixtures sharing the same *negative* group index never
        collide. We give every car fixture group ``-1`` to disable car-to-car
        contact, or group ``0`` (default category/mask) to enable it — in which
        case hulls collide with hulls/other wheels while wheel-wheel stays
        filtered out. Tiles are sensors and are unaffected either way.
        """
        group = 0 if enabled else -1
        bodies = [car.hull] + list(car.wheels)
        for body in bodies:
            for f in body.fixtures:
                filt = f.filterData
                filt.groupIndex = group
                f.filterData = filt

    # ------------------------------------------------------------------ track
    def _create_track(self):
        CHECKPOINTS = 12
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)
            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD
            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi
            while True:
                failed = True
                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break
                if not failed:
                    break
                alpha -= 2 * math.pi
                continue
            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x
            dest_dy = dest_y - y
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print(f"Track generation: {i1}..{i2} -> {i2 - i1}-tiles track")
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]
        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (x1 - TRACK_WIDTH * math.cos(beta1), y1 - TRACK_WIDTH * math.sin(beta1))
            road1_r = (x1 + TRACK_WIDTH * math.cos(beta1), y1 + TRACK_WIDTH * math.sin(beta1))
            road2_l = (x2 - TRACK_WIDTH * math.cos(beta2), y2 - TRACK_WIDTH * math.sin(beta2))
            road2_r = (x2 + TRACK_WIDTH * math.cos(beta2), y2 + TRACK_WIDTH * math.sin(beta2))
            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            # PER-CAR visit tracking (was a single bool in CarRacing).
            t.road_visited = [False] * self.num_agents
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (x1 + side * TRACK_WIDTH * math.cos(beta1), y1 + side * TRACK_WIDTH * math.sin(beta1))
                b1_r = (x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1), y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1))
                b2_l = (x2 + side * TRACK_WIDTH * math.cos(beta2), y2 + side * TRACK_WIDTH * math.sin(beta2))
                b2_r = (x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2), y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2))
                self.road_poly.append(
                    ([b1_l, b1_r, b2_r, b2_l], (255, 255, 255) if i % 2 == 0 else (255, 0, 0))
                )
        self.track = track
        return True

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._destroy()
        self.world.contactListener_bug_workaround = MultiFrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround

        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.tile_visited_count = [0] * self.num_agents
        self.car_done = np.zeros(self.num_agents, dtype=bool)
        self.lap_finished = np.zeros(self.num_agents, dtype=bool)
        self.new_lap = [False] * self.num_agents
        self.t = 0.0
        self._elapsed_steps = 0
        self.road_poly = []

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict) and "randomize" in options:
                randomize = options["randomize"]
            self._reinit_colors(randomize)

        while True:
            if self._create_track():
                break
            if self.verbose:
                print("retry to generate track (normal, happens occasionally)")

        poses = compute_spawn_poses(
            self.track,
            self.num_agents,
            TRACK_WIDTH,
            np_random=self.np_random if self.random_spawn else None,
        )
        self.cars = []
        for car_id, (angle, x, y) in enumerate(poses):
            car = Car(self.world, angle, x, y)
            car.hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]
            for w in car.wheels:
                w.car_id = car_id
            self._set_car_collisions(car, self.collisions)
            self.cars.append(car)

        if self.render_mode == "human":
            self._render("human")
        return self.step(None)[0], {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        if action is not None:
            action = np.asarray(action)
            if self.continuous:
                action = action.reshape(self.num_agents, 3).astype(np.float64)
            else:
                action = action.reshape(self.num_agents).astype(int)
            for k, car in enumerate(self.cars):
                if self.car_done[k]:
                    # Finished/crashed cars coast — no control input.
                    car.steer(0.0)
                    car.gas(0.0)
                    car.brake(0.0)
                    continue
                a = action[k]
                if self.continuous:
                    car.steer(-a[0])
                    car.gas(a[1])
                    car.brake(a[2])
                else:
                    act = int(a)
                    if not (0 <= act < 5):
                        raise InvalidAction(f"invalid discrete action {act}")
                    car.steer(-0.6 * (act == 1) + 0.6 * (act == 2))
                    car.gas(0.2 * (act == 3))
                    car.brake(0.8 * (act == 4))

        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.state = self._render("state_pixels")

        step_reward = np.zeros(self.num_agents)
        terminated = False
        truncated = False
        info = {}
        if action is not None:  # first call from reset() passes None
            self._elapsed_steps += 1
            active = ~self.car_done
            self.reward[active] -= 0.1
            for car in self.cars:
                car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward.copy()

            for k, car in enumerate(self.cars):
                if self.car_done[k]:
                    step_reward[k] = 0.0
                    continue
                if self.tile_visited_count[k] == len(self.track) or self.new_lap[k]:
                    self.car_done[k] = True
                    self.lap_finished[k] = True
                x, y = car.hull.position
                if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                    self.car_done[k] = True
                    self.lap_finished[k] = False
                    step_reward[k] = -100

            terminated = bool(np.all(self.car_done))
            if (
                self.max_episode_steps is not None
                and self._elapsed_steps >= self.max_episode_steps
            ):
                truncated = True
            info["terminated_per_car"] = self.car_done.copy()
            info["lap_finished_per_car"] = self.lap_finished.copy()

        if self.render_mode == "human":
            self._render("human")

        return (
            self.state,
            self._squeeze_reward(step_reward),
            terminated,
            truncated,
            info,
        )

    def _squeeze_reward(self, step_reward):
        """Single-car -> float (CarRacing parity); multi-car -> (N,) array."""
        return float(step_reward[0]) if self.num_agents == 1 else step_reward

    # ----------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "Calling render() without a render_mode. "
                f'Set it at init, e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        return self._render(self.render_mode)

    def _render(self, mode):
        assert mode in self.metadata["render_modes"]
        pygame.font.init()
        if "t" not in self.__dict__:
            return  # reset() not called yet

        if mode == "human":
            self._render_human()
            return self.isopen

        imgs = [self._render_car_image(k, mode) for k in range(self.num_agents)]
        # Single car -> a plain (H, W, 3) frame, exactly like CarRacing.
        return imgs[0] if self.num_agents == 1 else np.stack(imgs, axis=0)

    def _render_car_image(self, car_id, mode):
        surf = self._render_car_surface(
            car_id, draw_particles=(mode != "state_pixels")
        )
        size = (VIDEO_W, VIDEO_H) if mode == "rgb_array" else (STATE_W, STATE_H)
        return self._create_image_array(surf, size)

    def _render_car_surface(self, car_id, draw_particles=True):
        """Render the full 1000x800 egocentric view for one car (all cars drawn)."""
        car = self.cars[car_id]
        surf = pygame.Surface((WINDOW_W, WINDOW_H))

        # Use the hull heading (not a velocity-derived heading) so the image
        # stays in the distribution the single-car policy was trained on.
        angle = -car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -car.hull.position[0] * zoom
        scroll_y = -car.hull.position[1] * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self.surf = surf
        self._render_road(zoom, trans, angle)
        for c in self.cars:
            c.draw(surf, zoom, trans, angle, draw_particles)

        surf = pygame.transform.flip(surf, False, True)
        self.surf = surf
        self._render_indicators(car_id, WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render(
            f"{self.reward[car_id]:04.0f}", True, (255, 255, 255), (0, 0, 0)
        )
        rect = text.get_rect()
        rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        surf.blit(text, rect)
        return surf

    def _render_human(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            cols = math.ceil(math.sqrt(self.num_agents))
            rows = math.ceil(self.num_agents / cols)
            self._grid = (cols, rows)
            self._tile_size = (WINDOW_W // cols, WINDOW_H // rows)
            self.screen = pygame.display.set_mode(
                (self._tile_size[0] * cols, self._tile_size[1] * rows)
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

        cols, _ = self._grid
        tw, th = self._tile_size
        self.screen.fill(0)
        for k in range(self.num_agents):
            surf = self._render_car_surface(k, draw_particles=True)
            surf = pygame.transform.smoothscale(surf, (tw, th))
            r, c = k // cols, k % cols
            self.screen.blit(surf, (c * tw, r * th))
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [(bounds, bounds), (bounds, -bounds), (-bounds, -bounds), (-bounds, bounds)]
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(self.surf, poly, self.grass_color, zoom, translation, angle)
        for poly, color in self.road_poly:
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, car_id, W, H):
        car = self.cars[car_id]
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        true_speed = np.sqrt(
            np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1])
        )

        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
        render_if_min(car.wheels[0].omega, vertical_ind(7, 0.01 * car.wheels[0].omega), (0, 0, 255))
        render_if_min(car.wheels[1].omega, vertical_ind(8, 0.01 * car.wheels[1].omega), (0, 0, 255))
        render_if_min(car.wheels[2].omega, vertical_ind(9, 0.01 * car.wheels[2].omega), (51, 0, 255))
        render_if_min(car.wheels[3].omega, vertical_ind(10, 0.01 * car.wheels[3].omega), (51, 0, 255))
        render_if_min(
            car.wheels[0].joint.angle, horiz_ind(20, -10.0 * car.wheels[0].joint.angle), (0, 255, 0)
        )
        render_if_min(
            car.hull.angularVelocity, horiz_ind(30, -0.8 * car.hull.angularVelocity), (255, 0, 0)
        )

    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle, clip=True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly]
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2)
        )

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()
            self.screen = None
