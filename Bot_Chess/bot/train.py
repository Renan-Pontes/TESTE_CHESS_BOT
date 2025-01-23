# bot/train.py

import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

from bot.model import ChessModel
from bot.self_play import run_self_play
from bot.utils import process_self_play_data
from arena import arena_fight

def set_random_seed(seed):
    """Define uma semente global para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_model(
    total_iterations=1, 
    epochs_per_iteration=5, 
    total_games_selfplay=4, 
    num_workers=2, 
    arena_games=10, 
    arena_win_ratio=0.55,
    checkpoint_dir="checkpoints"
):
    """
    Fluxo completo de treinamento + arena:
      - Carrega (ou cria) um best_model
      - Cria candidate_model com os pesos de best_model
      - Gera dados de self-play com candidate_model
      - Treina candidate_model
      - Compara candidate_model vs best_model em arena
      - Se candidate_model for melhor, promove e salva
      - Repete para 'total_iterations' iterações

    Parâmetros:
    - total_iterations: número de iterações gerais (cada iteração faz self-play + treino + arena).
    - epochs_per_iteration: quantas épocas de treino para cada iteração.
    - total_games_selfplay: quantos jogos de self-play gerar em cada iteração.
    - num_workers: quantos processos paralelos para gerar self-play.
    - arena_games: quantos jogos a Arena usa para comparar candidate x best.
    - arena_win_ratio: % de vitórias necessária para promover o candidate (ex: 0.55 -> 55%).
    - checkpoint_dir: pasta para salvar checkpoints.

    """
    # Definindo semente para reprodutibilidade (opcional)
    set_random_seed(42)

    # Define dispositivo (GPU se disponível)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Treinando no dispositivo: {device}")

    # Caminho do melhor modelo
    best_model_file = "best_model.pth"

    # 1) Carregar ou criar best_model
    best_model = ChessModel()
    best_model.to(device)

    if os.path.exists(best_model_file):
        best_model.load_state_dict(torch.load(best_model_file, map_location=device))
        print("[INFO] best_model carregado com sucesso.")
    else:
        print("[WARN] best_model.pth não encontrado. Iniciando best_model do zero.")

    # Cria diretório para checkpoints se não existir
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Loop de iterações
    for iteration in range(1, total_iterations + 1):
        print(f"\n[TRAIN] ========== Início da iteração {iteration}/{total_iterations} ========== ")

        # 2) Cria candidate_model (cópia do best_model)
        candidate_model = ChessModel()
        candidate_model.to(device)
        candidate_model.load_state_dict(best_model.state_dict())

        # Define um otimizador (pode ajustar LR, etc.)
        optimizer = optim.Adam(candidate_model.parameters(), lr=1e-4)

        # 3) Gera dados via self-play
        print(f"[TRAIN] Gerando {total_games_selfplay} partidas de self-play ...")
        selfplay_data = run_self_play(candidate_model, total_games=total_games_selfplay, num_workers=num_workers)

        # 4) Treino (várias épocas)
        for epoch in range(1, epochs_per_iteration + 1):
            # Processa dados em tensores
            X, pi, z = process_self_play_data(selfplay_data)
            X = X.to(device)
            pi = pi.to(device)
            z = z.to(device)

            # Opcional: embaralhar os índices do dataset
            indices = torch.randperm(len(X))
            X = X[indices]
            pi = pi[indices]
            z = z[indices]

            # Treino em mini-lotes
            batch_size = 64
            total_batches = len(X) // batch_size
            running_loss = 0.0

            candidate_model.train()
            for b in range(total_batches):
                start = b * batch_size
                end = start + batch_size
                x_batch = X[start:end]
                pi_batch = pi[start:end]
                z_batch = z[start:end]

                optimizer.zero_grad()
                policy_out, value_out = candidate_model(x_batch)

                # CrossEntropy para policy, MSE para value
                # Notar: pi_batch deve ser "rótulos" ou "distribuições"?
                # Se pi_batch for OneHot, use algo como:
                loss_policy = F.cross_entropy(policy_out, pi_batch.argmax(dim=1))  

                loss_value = F.mse_loss(value_out, z_batch)
                loss = loss_policy + loss_value

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / total_batches if total_batches > 0 else 0
            print(f"[TRAIN] Iter {iteration}, Epoch {epoch}/{epochs_per_iteration}, Avg Loss = {avg_loss:.4f}")

            # Salva checkpoint a cada epoch
            checkpoint_path = os.path.join(checkpoint_dir, f"candidate_iter{iteration}_epoch{epoch}.pth")
            torch.save(candidate_model.state_dict(), checkpoint_path)
            print(f"[TRAIN] Checkpoint salvo em: {checkpoint_path}")

        # 5) Arena: candidate_model vs best_model
        print("[ARENA] Iniciando confronto candidate vs best ...")
        candidate_model.eval()
        best_model.eval()
        score_candidate, score_best = arena_fight(candidate_model, best_model, n_games=arena_games, simulations=50)
        print(f"[ARENA] Placares após {arena_games} jogos:\n"
              f" Candidate = {score_candidate}\n Best       = {score_best}")

        # 6) Critério de promoção
        if score_candidate > arena_win_ratio * (score_candidate + score_best):
            print("[ARENA] Candidate SUPEROU o Best! Promovendo para best_model.")
            torch.save(candidate_model.state_dict(), best_model_file)
            best_model.load_state_dict(candidate_model.state_dict())
        else:
            print("[ARENA] Best_model continua sendo melhor. Candidate descartado.")

        print(f"[TRAIN] ========== Fim da iteração {iteration}/{total_iterations} ==========\n")

    print("[TRAIN] Treinamento concluído!")


if __name__ == "__main__":
    # Parâmetros podem ser lidos via argparse ou config, por simplicidade estão fixos:
    train_model(
        total_iterations=2, 
        epochs_per_iteration=5, 
        total_games_selfplay=4, 
        num_workers=2, 
        arena_games=10, 
        arena_win_ratio=0.55,
        checkpoint_dir="checkpoints"
    )
