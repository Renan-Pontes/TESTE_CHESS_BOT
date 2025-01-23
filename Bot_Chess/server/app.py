import sys
import os

# Adiciona o diretório pai ao sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from bot.mcts import MCTS
from bot.model import ChessModel
import torch
import chess
import uuid
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# Carrega modelo treinado (se existir)
model = ChessModel()
try:
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    print("Modelo (best_model) carregado com sucesso.")
except:
    print("Aviso: 'best_model.pth' não encontrado. Usando modelo inicial.")

model.eval()

# Dicionário em memória para armazenar estados de cada partida
games_dict = {}  # { game_id: {"board": chess.Board(), ...}, ... }

@app.route("/")
def index():
    """
    Retorna uma página HTML simples com algumas instruções e botões de interação.
    Agora, usamos sessions para cada jogo.
    """
    html_content = """
    <html>
      <head>
        <title>Chess Bot</title>
      </head>
      <body>
        <h1>Bem-vindo ao Chess Bot (Multi-sessão)!</h1>
        <p>Exemplo de criação e interação com múltiplas partidas.</p>
        
        <button onclick="newGame()">Criar Novo Jogo</button>
        <br/><br/>
        
        <label>Game ID: </label>
        <input id="gameId" type="text" size="36" placeholder="Game ID aparece aqui após criar jogo"/>
        
        <br/><br/>
        <input id="moveInput" type="text" placeholder="Ex: e2e4"/>
        <button onclick="playerMove()">Enviar Lance</button>
        <br/><br/>
        <button onclick="botMove()">Lance do Bot</button>
        <br/><br/>
        <div id="status"></div>

        <script>
          function newGame() {
            fetch("/new_game", {method:"POST"})
             .then(r => r.json())
             .then(data => {
                document.getElementById("gameId").value = data.game_id;
                document.getElementById("status").innerHTML = "Novo jogo criado! FEN: " + data.fen;
             });
          }

          function playerMove() {
            const move = document.getElementById("moveInput").value;
            const game_id = document.getElementById("gameId").value;
            fetch("/player_move", {
              method:"POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ move_uci: move, game_id: game_id })
            })
            .then(r => r.json())
            .then(data => {
               if (data.error) {
                 document.getElementById("status").innerHTML = "Erro: " + data.error;
               } else {
                 document.getElementById("status").innerHTML = 
                   "Player move: " + move + "<br/>FEN: " + data.fen + "<br/>Status: " + data.status;
               }
            });
          }

          function botMove() {
            const game_id = document.getElementById("gameId").value;
            fetch("/bot_move", {
              method:"POST",
              headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ game_id: game_id })
            })
             .then(r => r.json())
             .then(data => {
                if (data.error) {
                  document.getElementById("status").innerHTML = "Erro: " + data.error;
                } else {
                  document.getElementById("status").innerHTML = 
                    "Bot move: " + data.bot_move + "<br/>FEN: " + data.fen + "<br/>Status: " + data.status;
                }
             });
          }
        </script>
      </body>
    </html>
    """
    return render_template_string(html_content)

@app.route("/new_game", methods=["POST"])
def new_game():
    """
    Cria um novo jogo, gera um UUID e armazena no dicionário 'games_dict'.
    """
    game_id = str(uuid.uuid4())
    board = chess.Board()
    games_dict[game_id] = {
        "board": board
    }
    return jsonify({
        "game_id": game_id,
        "fen": board.fen()
    })

@app.route("/player_move", methods=["POST"])
def player_move():
    """
    Recebe o lance do usuário (ex: e2e4) e o ID do jogo (game_id).
    """
    data = request.get_json()
    move_uci = data.get("move_uci", "")
    game_id = data.get("game_id", "")

    # Verifica se existe esse game_id
    if game_id not in games_dict:
        return jsonify({"error": "Game ID inválido!"}), 400

    board = games_dict[game_id]["board"]

    # Verifica se o lance é legal
    if move_uci not in [m.uci() for m in board.legal_moves]:
        return jsonify({"error": f"Lance inválido: {move_uci}"}), 400

    board.push_uci(move_uci)

    status_msg = get_board_status(board)
    return jsonify({
        "fen": board.fen(),
        "status": status_msg
    })

@app.route("/bot_move", methods=["POST"])
def bot_move():
    """
    Calcula e faz o lance do bot no tabuleiro do game_id.
    """
    data = request.get_json()
    game_id = data.get("game_id", "")

    if game_id not in games_dict:
        return jsonify({"error": "Game ID inválido!"}), 400

    board = games_dict[game_id]["board"]

    # Caso o jogo já esteja finalizado
    if board.is_game_over():
        return jsonify({"error": "Este jogo já acabou.", "fen": board.fen()}), 400

    mcts = MCTS(model, simulations=50)  # exemplo: 50 simulações
    best_move = mcts.get_best_move(board)
    if not best_move:
        return jsonify({"error": "Sem movimentos disponíveis."}), 400

    board.push(best_move)
    status_msg = get_board_status(board)

    return jsonify({
        "bot_move": best_move.uci(),
        "fen": board.fen(),
        "status": status_msg
    })

def get_board_status(board):
    """
    Retorna uma string simples indicando status (continua, xeque-mate, empate, etc).
    """
    if board.is_checkmate():
        return "Checkmate!"
    elif board.is_stalemate():
        return "Empate por afogamento (Stalemate)."
    elif board.is_insufficient_material():
        return "Empate por falta de material."
    elif board.is_game_over():
        return "Partida finalizada."
    else:
        return "Jogo em andamento."

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
