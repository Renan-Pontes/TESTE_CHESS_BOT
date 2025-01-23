# arena.py

import chess
import random
from bot.mcts import MCTS

def arena_fight(modelA, modelB, n_games=10, simulations=100):
    """
    Faz 'n_games' partidas entre modelA e modelB.
    - Metade das partidas, modelA joga de brancas, modelB de pretas.
    - Metade invertido.
    - Retorna (scoreA, scoreB).

    :param modelA: instância do ChessModel (PyTorch) ou similar
    :param modelB: instância do ChessModel
    :param n_games: quantidade total de partidas (idealmente par)
    :param simulations: número de simulações que cada MCTS deve realizar
    """
    scoreA = 0.0
    scoreB = 0.0
    games_as_white = n_games // 2
    games_as_black = n_games - games_as_white  # caso n_games seja ímpar
    
    # Partidas onde A é branco, B é preto
    for _ in range(games_as_white):
        result = play_game(modelA, modelB, simulations=simulations, white_is_modelA=True)
        if result == 1:
            scoreA += 1.0
        elif result == -1:
            scoreB += 1.0
        else:
            # empate
            scoreA += 0.5
            scoreB += 0.5
    
    # Partidas onde A é preto, B é branco
    for _ in range(games_as_black):
        result = play_game(modelB, modelA, simulations=simulations, white_is_modelA=True)
        # Nesse caso, 'play_game(modelB, modelA, ...)' retorna 1 se modelB (brancas) vencer
        # mas modelB aqui é "A" como white, e modelA é "B" como black. Então precisamos
        # interpretar corretamente:
        if result == 1:
            # modelB (que na verdade é o A original) ganhou
            scoreB += 1.0
        elif result == -1:
            # modelA (que aqui é o B da função) ganhou
            scoreA += 1.0
        else:
            scoreA += 0.5
            scoreB += 0.5

    return scoreA, scoreB

def play_game(model_white, model_black, simulations=100, white_is_modelA=True):
    """
    Roda UMA partida entre:
      - model_white jogando de brancas
      - model_black jogando de pretas
    Retorna:
      1 se as brancas venceram
      -1 se as pretas venceram
      0 se foi empate
    """
    board = chess.Board()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = MCTS(model_white, simulations=simulations).get_best_move(board)
        else:
            move = MCTS(model_black, simulations=simulations).get_best_move(board)
        if not move:
            # Se não existir movimento legal (erro ou bug?), encerra
            break
        board.push(move)

    # Verificando resultado final:
    if board.is_checkmate():
        # Quem deu mate?
        return 1 if board.turn == chess.BLACK else -1
        # Importante notar:
        # 'board.turn == chess.BLACK' significa que as pretas acabaram de jogar?
        # Por convenção do python-chess, se `board.is_checkmate()` e `board.turn == chess.BLACK`,
        # significa que as pretas estão para jogar, ou seja, as brancas deram mate na jogada anterior.
    elif board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
        return 0
    else:
        # Outras formas de game_over podem surgir (ex: 50-move rule).
        return 0
