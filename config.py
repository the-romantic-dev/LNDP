from dataclasses import dataclass

PROJECT = "lndp"


@dataclass
class Config:
    project: str = PROJECT  # имя wandb проекта
    seed: int = 1
    n_seeds: int = 1
    # --- trainer ---
    strategy: str = "CMA_ES"
    popsize: int = 256
    generations: int = 10
    ckpt_file: str = ""
    log: int = 0  # 1 - логгинг в wandb
    eval_reps: int = 1  # number of evaluations to average over
    # --- task ---
    env_name: str = "CartPole-v1"
    n_episodes: int = 3  # число эпизодов
    l1_penalty: float = 0.
    dev_after_episode: int = 0
    env_size: int = 5
    p_switch: float = 0.
    dense_reward: int = 0
    # --- model ---
    n_nodes: int = 32  # максимальное число нод в сети
    node_features: int = 8
    edge_features: int = 4
    pruning: int = 1
    synaptogenesis: int = 1
    rnn_iters: int = 3  # number of propagation steps
    dev_steps: int = 0  # number of developmental steps
    p_hh: float = 0.1  # initial connection probabilities (avergae/variance)
    s_hh: float = 0.0001
    p_ih: float = 0.1
    s_ih: float = 0.0001
    p_ho: float = 0.1
    s_ho: float = 0.0001
    use_bias: int = 0
    is_recurrent: int = 0  # activity is resetted between env steps if is_recurrent=0
    gnn_iters: int = 1  # number of GNN forward pass
    stochastic_decisions: int = 0  # if synaptogenesis or pruning is probabilistic
    block_lt_updates: int = 0  # if 1 will block any change during the agent lifetime
