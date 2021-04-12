{
    "name": "Garaham", 
    "n_gpu": 1,
    "data_dir": "data/full_df.csv",

    "arch": {
        "type": "Odr_model",
        "args": {}
    },

    "data_loader": {
        "type": "odr_data_loader", 
        "args":{
            "data_dir": "data/full_df.csv",
            "batch_size": 25,
            "shuffle": true,
            "validation_split": 0.3,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args":{
            "lr": 0.01,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "nll_loss",
    "metrics": [
        "accuracy", "top_k_acc"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 2,

        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        
        "monitor": "min val_loss",
        "early_stop": 10,

        "tensorboard": true
    }
}