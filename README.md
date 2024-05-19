## **DPO-deepseekmoe-test**
- JINIAC DeepseekMoEのDPOのテストコード


### **1. 必要なライブラリのインストール**
-  以下に従い、環境構築を行う
    -  https://github.com/matsuolab/ucllm_nedo_prod/blob/main/train/README-train_gcp_play_single_node_multi_gpu.md

- トレーニングに必要なライブラリを追加でインストール。
    
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
https://github.com/hibikaze-git/sft-deepseekmoe-test  
https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.mdx#multi-gpu-training
https://huggingface.co/docs/transformers/en/perf_train_gpu_many?select-gpu=Accelerate#number-of-gpus

コードは以下を参考にしております。  
https://note.com/npaka/n/n23576a1211a0
