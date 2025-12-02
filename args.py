from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Args:
    coach_args: dict = {
        "num_Iters": 10,
        "num_episodes": 100,
        'tempThreshold': 12, #12 # 前 n 步 不一定挑最好的走步
        'updateThreshold': 0.6, # 超過
        'maxlenOfQueue': 200000,#200000 訓練記錄
        'oneNet': True,
        'numItersForTrainExamplesHistory': 10,
        'roundLimit': 150,

        'numSelfPlayPool': 4,
        'numTestPlayPool': 12,
        'numPerProcessPlay': 4,
    }

    nn_args: dict = {
        "lr": 0.001,
        "drop_out": 0.3,
        "epoch": 10,
        "batch_size": 64,
        'num_channels': 64, #512
        'num_blocks': 5
    }
    
    file_args: dict = {

    }