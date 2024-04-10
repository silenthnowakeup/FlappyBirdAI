import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNModel(nn.Module):
    def __init__(self, n_action=2):
        super(DQNModel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, n_action)
        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        output = self.out(x)
        return output


class DQN:
    def __init__(self, n_action, lr=1e-6):
        self.criterion = torch.nn.MSELoss()
        self.model = DQNModel(n_action).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        """
        Вычисляет значения Q-функции для всех действий в заданном
         состоянии, применяя обученную модель
        :param s: входное состояние
        :return: значения Q для всех действий в состоянии s
        """
        return self.model(torch.Tensor(s).to(device))  # Переносим на нужное устройство

    def update(self, y_predict, y_target):
        """
        Обновляет веса DQN на основе обучающего примера
        :param y_predict: предсказанное значение
        :param y_target: целевое значение
        :return: потеря
        """
        loss = self.criterion(y_predict, y_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def replay(self, memory, replay_size, gamma):
        """
        Воспроизведение опыта
        :param memory: буфер воспроизведения опыта
        :param replay_size: сколько примеров использовать при каждом
         обновлении модели
        :param gamma: коэффицент обесценивания
        :return: потеря
        """
        if len(memory) >= replay_size:
            replay_data = random.sample(memory, replay_size)
            state_batch, action_batch, next_state_batch, \
                reward_batch, done_batch = zip(*replay_data)
            state_batch = torch.cat(
                tuple(state for state in state_batch)).to(device)  # Переносим на нужное устройство
            next_state_batch = torch.cat(
                tuple(state for state in next_state_batch)).to(device)  # Переносим на нужное устройство
            q_values_batch = self.predict(state_batch)
            q_values_next_batch = self.predict(next_state_batch)
            reward_batch = torch.from_numpy(np.array(
                reward_batch, dtype=np.float32)[:, None]).to(device)  # Переносим на нужное устройство
            action_batch = torch.from_numpy(
                np.array([[1, 0] if action == 0 else [0, 1]
                          for action in action_batch], dtype=np.float32)).to(device)  # Переносим на нужное устройство
            q_value = torch.sum(q_values_batch * action_batch, dim=1)
            td_targets = torch.cat(
                tuple(reward if terminal else reward +
                      gamma * torch.max(prediction) for
                      reward, terminal, prediction
                      in zip(reward_batch, done_batch,
                             q_values_next_batch))).to(device)  # Переносим на нужное устройство
            loss = self.update(q_value, td_targets)
            return loss

