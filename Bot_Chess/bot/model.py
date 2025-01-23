# bot/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Bloco residual simples: 
      - Conv2d -> BN -> ReLU -> Conv2d -> BN
      - Skip connection + ReLU
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # Skip Connection
        out += residual
        out = F.relu(out)
        return out

class ChessModel(nn.Module):
    """
    Modelo inspirado no AlphaZero:
    - Convoluções iniciais (input conv)
    - Vários blocos residuais
    - Head de Política (Policy Head)
    - Head de Valor (Value Head)
    """
    def __init__(self, in_channels=14, num_channels=64, num_res_blocks=3, num_moves=4672):
        """
        :param in_channels: número de planos de entrada (ex: 14)
        :param num_channels: número de filtros de cada conv
        :param num_res_blocks: quantos blocos residuais
        :param num_moves: tamanho da saída de política (número máximo de movimentos possíveis)
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.num_moves = num_moves

        # 1) Convolução inicial
        self.conv_in = nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(num_channels)

        # 2) Blocos residuais
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 3) Cabeça de Política (Policy Head)
        #    Conv de 1x1 -> BN -> ReLU -> Flatten -> Linear -> Softmax (normalmente no forward)
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, num_moves)  # 8x8 -> 64 posições, 2 canais -> 128

        # 4) Cabeça de Valor (Value Head)
        #    Conv de 1x1 -> BN -> ReLU -> Flatten -> Linear -> ReLU -> Linear -> Tanh
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 128)   # 8x8 -> 64, mas flatten -> 64
        self.value_fc2 = nn.Linear(128, 1)       # valor escalar

    def forward(self, x):
        """
        :param x: tensor de shape (batch_size, in_channels, 8, 8)
        :return: (policy_logits, value) 
          policy_logits -> (batch_size, num_moves)
          value         -> (batch_size, 1) no range (-1, +1)
        """
        # Normalmente, x seria [N, 14, 8, 8] (ou outra contagem de planos)

        # Convolução inicial
        out = self.conv_in(x)
        out = self.bn_in(out)
        out = F.relu(out)

        # Blocos residuais
        for block in self.res_blocks:
            out = block(out)

        # Policy Head
        p = self.policy_conv(out)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)  # flatten [batch_size, 2, 8, 8] -> [batch_size, 2*8*8]
        p = self.policy_fc(p)      # shape: [batch_size, num_moves]
        # Em treinamento, podemos aplicar CrossEntropy, que internamente faz log_softmax.
        # Em inferência, podemos aplicar F.softmax(p, dim=1) para obter prob.

        # Value Head
        v = self.value_conv(out)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)  # flatten [batch_size, 1, 8, 8] -> [batch_size, 64]
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        v = torch.tanh(v)  # valor no range [-1, 1]

        return p, v
