"""Competitive tile-visit reward (pure function).

Mirrors the MultiCarRacing scheme: the first car to reach a tile gets the full
``1000 / num_tiles`` bonus; each later car to reach the same tile gets a share
damped by how many cars already beat it there. With two cars, the first to a
tile earns ``1000 / N_tiles`` and the second ``500 / N_tiles``.

The point of the damping is to keep the reward dense enough to learn basic
driving (every fresh tile still pays) while turning "get to track real estate
first" into an explicit incentive — the seed of racecraft. The single-car
CarRacing reward is the special case ``num_agents == 1`` (always the full bonus).
"""


def competitive_tile_reward(num_agents, num_tiles, past_visitors):
    """Reward for a car visiting a tile for the first time.

    Args:
        num_agents: total cars in the race.
        num_tiles: total tiles in the track.
        past_visitors: how many *other* cars already visited this tile
            (0 for the first car to arrive).
    """
    return (num_agents - past_visitors) / num_agents * (1000.0 / num_tiles)
