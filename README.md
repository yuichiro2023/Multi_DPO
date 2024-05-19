### **1. 必要なライブラリのインストール**

- トレーニングに必要なライブラリ（**`transformers`**, **`datasets`**, **`torch`**, **`accelerate`** など）をインストール。
    
    ```bash
    pip install -r requirements.txt
    ```
    
### **2. スクリプトの実行**

- configファイルが保存されているディレクトリに移動し、**`accelerate launch`** コマンドを使用してスクリプトを実行します。
    
    ```bash
    accelerate launch --config_file accelerate_config.yaml DPO/Multi_DPO_prod.py
    ```
    
    このコマンドは **`accelerate`** によって管理され、configによって指定された数のプロセスがGPUに割り当てられます。
    
    このプロセスにより、指定された数のGPUを効率的に使用し、トレーニングの効率を最大化できます。また、**`accelerate`** の設定がスクリプトに適切に反映されるようになり、GPUリソースの管理が簡素化されます。

### **参考資料**
上記の記載は以下資料を参考にしています。
https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.mdx#multi-gpu-training
https://huggingface.co/docs/transformers/en/perf_train_gpu_many?select-gpu=Accelerate#number-of-gpus
