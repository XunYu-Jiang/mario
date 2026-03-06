import torch
class Args:
    COACH_ARGS: dict = {
        "num_iters": 2,
        "num_episodes": 1,
        'tempThreshold': 12, #12 # 前 n 步 不一定挑最好的走步
        'updateThreshold': 0.6, # 超過
        'buffer_size': 20000,#200000 訓練記錄
        'oneNet': True,
        'numItersForTrainExamplesHistory': 10,
        'roundLimit': 150,

        'numSelfPlayPool': 4,
        'numTestPlayPool': 12,
        'numPerProcessPlay': 4,
    }

    NN_ARGS: dict = {
        "drop_out": 0.3,
        'num_channels': 64, #512
        'num_blocks': 5,
    }

    TRAIN_ARGS: dict = {
        "lr": 1e-3,
        "epoch": 10,
        "batch_size": 32,
        'q_learning_discount': 0.9,  #discount in (last_reward + "gamma" * value_pred) - last_value_pred
        "device": "cuda"        
        # "device": "cpu"        

    }
    
    FILE_ARGS: dict = {
        "log_dir": "./logs/",
        "log_file": "log.txt"
    }

    EX_REPLAY: dict = {
        "batch_size": 32,
        "buffer_size": 1000
    }