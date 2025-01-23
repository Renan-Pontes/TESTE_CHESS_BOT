# bot/utils.py

import json
import torch

def save_game_data(games_data, filepath):
    """
    Salva a lista de dados de partidas em um arquivo .json
    """
    with open(filepath, 'w') as f:
        json.dump(games_data, f, indent=2)

def process_self_play_data(games_data):
    """
    Transforma games_data em tensores X (posições), pi (prob lances), z (resultados).
    Aqui apenas devolvemos tensores vazios para exemplo.
    """
    # Exemplo: sem fazer nada de útil de fato
    X = torch.rand(10, 14, 8, 8)   # 10 exemplos, input 14x8x8
    pi = torch.zeros(10, 4672)     # 10 dist. de movimento
    z = torch.zeros(10, 1)         # 10 resultados
    return X, pi, z
