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

### **参考資料**
上記の記載は以下資料を参考にしています。
https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.mdx#multi-gpu-training
https://huggingface.co/docs/transformers/en/perf_train_gpu_many?select-gpu=Accelerate#number-of-gpus
