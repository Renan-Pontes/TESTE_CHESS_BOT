# bot/mcts.py

import chess
import random
import math
import copy

class MCTSNode:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0  # se quisermos usar rede para policy
        self.is_terminal = board.is_game_over()

    @property
    def value(self):
        # valor médio
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

class MCTS:
    def __init__(self, model=None, simulations=100, c_puct=1.0):
        """
        model: Rede neural para policy/value
        simulations: número de simulações
        c_puct: constante de exploração
        """
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct

    def get_best_move(self, board):
        """
        Executa MCTS por 'self.simulations' vezes, retorna o melhor lance.
        """
        if board.is_game_over():
            return None

        root_node = MCTSNode(board)

        for _ in range(self.simulations):
            node = self._selection(root_node)
            if not node.is_terminal:
                node = self._expand(node)
            reward = self._simulation(node.board)
            self._backpropagate(node, reward)

        # após as simulações, escolher o filho com maior número de visitas
        best_child = max(root_node.children.values(), key=lambda n: n.visits if n else 0)
        # para descobrir qual foi o lance associado, vamos inverso: 
        best_move = None
        for move, child_node in root_node.children.items():
            if child_node is best_child:
                best_move = move
                break
        return best_move

    def _selection(self, node):
        """
        Navega pela árvore até encontrar um nó não totalmente expandido ou terminal
        usando a UCB formula.
        """
        while node.children and not node.is_terminal:
            # todos filhos já expandidos, selecionar via UCB
            node = self._select_child(node)
        return node

    def _select_child(self, node):
        best_move = None
        best_value = -float('inf')
        best_child = None

        for move, child in node.children.items():
            # UCB = Q + U = child.value + c_puct * sqrt(ln(parent.visits)/(1+child.visits))
            q = child.value
            u = self.c_puct * math.sqrt(node.visits + 1) / (1 + child.visits)
            value = q + u
            if value > best_value:
                best_value = value
                best_move = move
                best_child = child
        return best_child

    def _expand(self, node):
        """
        Cria filhos para todos os lances legais do 'node.board' (caso ainda não existam).
        Retorna UM filho (poderia retornar random).
        """
        if node.is_terminal:
            return node

        legal_moves = list(node.board.legal_moves)
        for move in legal_moves:
            if move not in node.children:
                new_board = copy.deepcopy(node.board)
                new_board.push(move)
                node.children[move] = MCTSNode(new_board, parent=node)
        # Escolher um dos filhos para simulação
        return random.choice(list(node.children.values()))

    def _simulation(self, board):
        """
        Simulação simples até o final ou um certo depth. 
        (Pode ser substituída pelo uso do 'model' para estimar valor ou escolher lances).
        """
        sim_board = copy.deepcopy(board)
        # Joga random até o final
        while not sim_board.is_game_over():
            moves = list(sim_board.legal_moves)
            move = random.choice(moves)
            sim_board.push(move)

        # Retorna +1 se white ganhou, -1 se black ganhou, 0 se empate.
        if sim_board.is_checkmate():
            return 1 if sim_board.turn == chess.BLACK else -1
        return 0

    def _backpropagate(self, node, reward):
        """
        Sobe na árvore atualizando 'visits' e 'value_sum'.
        """
        while node is not None:
            node.visits += 1
            node.value_sum += reward
            node = node.parent
            # Se o reward é 1 para brancas, deve inverter se mudarmos de perspectiva?
            # Esse é um ponto de implementação que pode ficar mais complexo
            # se considerarmos a perspectiva do jogador. Aqui fica simplificado.
