# MPCutter
a tool for the prediction of cleavage sites by MP-BERT(pre-trained language model)

## 1、Feature
• Predicts protein cleavage sites with high accuracy using the MP-BERT model.<br>
• Utilizes the power of the MindSpore framework and Ascend 910 NPU for efficient inference.<br>

## 2、<span style="background-color: yellow;">Train MPCutter</span> 
### Required configuration and environment
|| &nbsp;PYTHON=3.7&nbsp;|| <font color="blue">MINDSPORE=1.8</font> || <font color="green">DOCKER>=18.03</font> || <font color="purple">UniRef50 2048 base for MP-BERT</font> || <font color="orange">	<br>
•	Ensure you have installed the necessary firmware and drivers for Huawei Atlas server (Linux operating system, Huawei Ascend 910 NPU). <br>
•	Running MP-BERT with the MindSpore framework on Ascend, it is recommended to use Docker for rapid deployment.<br>
•	Confirm the installation of Ubuntu 18.04 or CentOS 7.6 64-bit operating system based on ARM architecture, running on an Nvidia GPU server. The server supports GPU and can install environments such as Docker, Conda, and Pip. <br>
•   Fine-tuning of MP-BERT can be achieved on an NPU or GPU card. Load the pretrained model according to your needs for fine-tuning. <br>
•   For more information, please refer to the original author of MP-BERT: https://github.com/BRITian/MP-BERT

 ### Training
•	Data processing: The collated data set is converted into a format that MP-BERT can recognize       <br>
              ```python
/generate_for_finetune/generate_seq_for_classification_1x.py --data_dir <input_dataset> --output_dir <output_datset> --vocab_file /generate_for_finetune/vocab_v2.txt --max_seq_length 2048
              ```  <br>  <br>
•   Finetune:          We added a classification layer on top of MP-BERT. The last layer’s [CLS] feature is processed through a fully connected network and SoftMax to output predictions, facilitating the task of site prediction classification.  <br> 
              ``` python
/finetune_network/mpbert_classification.py --config_path /finetune_network/config_2048.yaml --do_train True --do_eval True --description correlation --num_class 2 --epoch_num <num> --frozen_bert False --device_id <device_target> --data_url <input_dataset> --load_checkpoint_url <MP-BERT_models> --output_url <saved_models> --task_name test --train_batch_size 32 
              ```    <br>  <br>
•   Technical route:   Please refer to the raw article

## 3、Usage
only need to use the trained model to predict your own target data   <br>
              ``` python
/finetune_network/mpbert_classification.py --config_path /finetune_network/config_2048.yaml --do_predict True --data_url <predicted_data> --description correlation --load_checkpoint_url <your_models> --output_url <output_pathway> --vocab_file /finetune_network/vocab_v2.txt --return_csv True --device_id <device_target>
              ```      <br>  <br> 
