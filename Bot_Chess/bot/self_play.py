# bot/self_play.py

import multiprocessing as mp
import chess
import random
from .mcts import MCTS
from .utils import save_game_data

def self_play_worker(model, games_to_play, return_dict, worker_id):
    games_data = []
    for _ in range(games_to_play):
        board = chess.Board()
        moves = []
        while not board.is_game_over():
            mcts = MCTS(model)
            move = mcts.get_best_move(board)
            board.push(move)
            moves.append(move.uci())
        # Guardar a lista de lances para exemplificar
        games_data.append(moves)
    return_dict[worker_id] = games_data

def run_self_play(model, total_games=10, num_workers=2):
    manager = mp.Manager()
    return_dict = manager.dict()
    jobs = []
    games_per_worker = total_games // num_workers

    for worker_id in range(num_workers):
        p = mp.Process(target=self_play_worker,
                       args=(model, games_per_worker, return_dict, worker_id))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    # Combina tudo
    all_data = []
    for worker_id in range(num_workers):
        all_data.extend(return_dict[worker_id])

    # Você pode salvar ou retornar para processamento
    save_game_data(all_data, "selfplay_data.json")

    return all_data
