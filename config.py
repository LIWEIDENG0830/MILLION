from dataclasses import dataclass

@dataclass
class Config:

    ''' Dataset Settings '''
    n_stocks: int = -1
    state_dim: int = -1
    window: int = 20
    # NAS100
    # data_dir: str = './datasets/NAS100/'
    # train_start_date: str = '2009-01-01'
    # train_end_date: str = '2019-01-01'
    # valid_start_date: str = '2019-01-01'
    # valid_end_date: str = '2020-01-01'
    # test_start_date: str = '2020-01-01'
    # test_end_date: str = '2021-01-01'
    # DOW30
    # data_dir: str = './datasets/DOW30/'
    # train_start_date: str = '2009-01-01'
    # train_end_date: str = '2018-01-01'
    # valid_start_date: str = '2018-01-01'
    # valid_end_date: str = '2019-01-01'
    # test_start_date: str = '2019-01-01'
    # test_end_date: str = '2020-01-01'
    # CRYPTO10
    # 2018-08-02 2023-12-31
    data_dir: str = './datasets/CRYPTO10DAILY/'
    train_start_date: str = '2018-08-22'
    train_end_date: str = '2021-01-01'
    valid_start_date: str = '2021-01-01'
    valid_end_date: str = '2021-07-01'
    test_start_date: str = '2021-07-01'
    test_end_date: str = '2023-10-31'
    dataset: str = ''                               # Placeholder
    npz_file: str = 'processed.npz'

    ''' Training Settings '''
    seed: int = 0
    device: str = 'cuda:2'
    optimizer: str = 'AdamW'

    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 100
    eval_interval: int = 1
    plot_interval: int = 10
    result_save_dir: str = './results/'
    target: str = 'downsiderisk'
    epochs: int = 200

    lookback: int = 20
    cov_weight: float = 10
    cov_alpha: float = 0.5

    mse_alpha: float = 0
    ranking_alpha: float = 0

    alpha_downside: float = 0.005         # for downsiderisk
    action_constraint: float = 0.2      # for actionconstraint
    action_constraint_weight: float = 1 # for actionconstraint

    # action optimize
    n_optimize: int = 5
    predefined_cov_threshold: float = 0

    n_quantile: int = 3

    ''' Model Settings '''
    save_dir: str = './save/'
    # For NAS100
    net_dims: str = '64,32'

    # model_name: str = 'LSTMHADW'
    model_name: str = 'LSTMHASCDW'
    trans_cost: bool = False

    temporature: float = 1
    n_head: int = 4

    trans_rate: float = 0.005