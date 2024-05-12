### **1. 必要なライブラリのインストール**

- トレーニングに必要なライブラリ（**`transformers`**, **`datasets`**, **`torch`**, **`accelerate`** など）をインストール。
    
    ```bash
    pip install -r requirements.txt
    ```
    
### **2. 環境設定**

- **`accelerate config`** コマンドを使用して、マルチGPU設定を行います。この設定はトレーニングのパフォーマンスやGPUの使用方法に大きく影響します。
    
    ```bash
    accelerate config
    ```
    
    このステップでは、使用するGPUの数や、混合精度計算の使用、分散トレーニングの種類（DDPなど）に関する質問に答えます。
    
    - Configの設定
        
        ### **1. How many processes would you like to use?**
        
        この質問は、利用したいプロセス（GPU）の数を尋ねます。2つのGPUを使用したい場合は、「2」と答えます。
        
        ```bash
        Number of processes: 2
        ```
        
        ### **2. Mixed precision**
        
        混合精度を使用するかどうかの質問です。bf16を使用する場合、「Yes」を選択し、特にbfloat16を指定するオプションがあればそれを選びます。**`accelerate`**がこのオプションを直接サポートしていない場合、設定後にスクリプト内で明示的に設定する必要があります。
        
        ```bash
        Do you want to use mixed precision (fp16 or bf16)? [Yes/No]: Yes
        ```
        
        ### **3. Which type of distributed computing to use?**
        
        分散コンピューティングのタイプを選択します。一般的には、マルチGPU設定には**`DistributedDataParallel (DDP)`**が推奨されます。
        
        ```bash
        bashCopy code
        Which type of distributed computing do you wish to use? [NO / DDP / DP / ...]: DDP
        ```
        
        ### **4. Where should your logs be stored?**
        
        ログを保存する場所を指定します。これは好みによりますが、結果を追跡しやすいディレクトリを指定します。
        
        ```bash
        Where do you want to store the logs?: ./train_logs
        ```
        

### **3. スクリプトの実行**

- スクリプトが保存されているディレクトリに移動し、**`accelerate launch`** コマンドを使用してスクリプトを実行します。**`-num_processes`** オプションで使用するGPUの数を指定します。
    
    ```bash
    accelerate launch --num_processes 2 {script_name.py}
    ```
    
    このコマンドは **`accelerate`** によって管理され、指定された数のプロセス（この場合は2つ）がGPUに割り当てられます。
    

### **4.プログラムでのGPU割り当て**

- **`PartialState`** クラスを使用して、各プロセスがどのGPUにアクセスするかを制御します。これにより、スクリプト内で **`accelerate`** の設定が反映され、各GPUが適切に利用されます。
    
    ```python
    pythonCopy code
    from accelerate import PartialState
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        ...,
        device_map={'':device_string}
    )
    
    ```
    
    ここで、**`device_map`** に **`PartialState().process_index`** を設定することで、**`accelerate`** によって割り当てられた具体的なGPUをモデルが使用するように指示しています。
    このプロセスにより、指定された数のGPUを効率的に使用し、トレーニングの効率を最大化できます。また、**`accelerate`** の設定がスクリプトに適切に反映されるようになり、GPUリソースの管理が簡素化されます。

### **参考資料**
上記の記載は以下資料を参考にしています。
https://github.com/huggingface/trl/blob/main/docs/source/sft_trainer.mdx#multi-gpu-training
https://huggingface.co/docs/transformers/en/perf_train_gpu_many?select-gpu=Accelerate#number-of-gpus
