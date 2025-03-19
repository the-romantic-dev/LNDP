import pickle
import equinox as eqx

import gym

env = gym.make('Acrobot-v1')
state = env.reset()
done = False
model = eqx.lo
with open("ckpts/Acrobot-v1_1_config.eqx", "rb") as f:
    model = pickle.load(f)

while not done:
    env.render()
    action = model.predict(state)  # Получаем действие от модели
    state, reward, done, info = env.step(action)
env.close()
