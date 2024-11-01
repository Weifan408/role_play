import gymnasium as gym


class MaxStepWrapper(gym.core.Wrapper):
    def __init__(self, env, max_steps):
        super().__init__(env)
        self.max_steps = max_steps
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.steps += 1
        obs, rew, done, _, info = self.env.step(action)
        done["__all__"] = False
        if self.steps >= self.max_steps:
            done["__all__"] = True
        return obs, rew, done, _, info
