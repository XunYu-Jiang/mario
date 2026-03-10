import torch
class Args:
    COACH_ARGS: dict = {
        "num_iters": 100,   #100
        "num_episodes": 50, #50
        "is_multiprocess": False,   
        'tempThreshold': 12 #12 # 前 n 步 不一定挑最好的走步
    }

    NN_ARGS: dict = {
        "drop_out": 0.3,
        'num_channels': 64, #512
        'num_blocks': 5,
    }

    TRAIN_ARGS: dict = {
        "lr": 1e-3,
        "batch_size": 32,
        "buffer_size": 250,
        'q_learning_discount': 0.9,  #discount in (last_reward + "gamma" * value_pred) - last_value_pred
        "device": "cuda"        
        # "device": "cpu"        

    }
    
    FILE_ARGS: dict = {
        "log_dir": "./logs/",
        "log_file": "log.txt",
        "model_dir": "./models/",
        "vod_dir": "./vods/"
    }