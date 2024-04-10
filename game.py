import torch
from environment import FlappyBird, screen_width
from utils import pre_processing

saved_path = 'trained_models'
image_size = 84
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("{}/final".format(saved_path), map_location=device)
n_episode = 100
for episode in range(n_episode):
    env = FlappyBird()
    image, reward, is_done = env.next_step(0)
    image = pre_processing(image[:screen_width, :int(env.base_y)], image_size, image_size)
    image = torch.from_numpy(image).to(device)
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :].to(device)
    while True:
        prediction = model(state)[0]
        action = torch.argmax(prediction).item()
        next_image, reward, is_done = env.next_step(action)
        next_image = pre_processing(next_image[:screen_width, :int(env.base_y)], image_size, image_size)
        next_image = torch.from_numpy(next_image).to(device)
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :].to(device)
        state = next_state
