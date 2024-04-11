# MPCutter
a tool for the prediction of cleavage sites by MP-BERT(pre-trained language model)

## 1、Feature
• Predicts protein cleavage sites with high accuracy using the MP-BERT model.<br>
• Utilizes the power of the MindSpore framework and Ascend 910 NPU for efficient inference.<br>

## 2、<span style="background-color: yellow;">Train MPCutter</span> 
### Required configuration and environment
||   PYTHON=3.7   ||   MINDSPORE=1.8   ||   DOCKER>=18.03   ||   UniRef50 2048 base for MP-BERT   ||	<br><br>
•	Ensure that you have properly installed the required firmware and drivers tailored for the Huawei Atlas server (running on the Linux operating system and equipped with the Huawei Ascend 910 NPU). <br>
•	When running MP-BERT utilizing the MindSpore framework on Ascend, it is highly recommended to utilize Docker for swift and efficient deployment.<br>
•	Furthermore, please verify the installation of either Ubuntu 18.04 or CentOS 7.6, both 64-bit operating systems based on ARM architecture, ideally on an Nvidia GPU server. This server is designed to support GPU operations and allows for the installation of various environments, such as. Docker, Conda, and Pip. <br>
• Fine-tuning of MP-BERT can be conducted either on an NPU or GPU card. Load the pretrained model based on your requirements for fine-tuning. <br>
• For further details, please consult the original author of MP-BERT at: [BRITian/MP-BERT](https://github.com/BRITian/MP-BERT).

 ### Training
•	Your dataset should be organized in CSV format, with the following data structure.
| Substrate_id | sliding_windows | label |
|---------|---------|---------|
| Q9Y490	 | LSLKISIGNVVKTMQFEP | 1 |
| ... | ... | ... |

•	Data processing: The collated data set is converted into a format that MP-BERT can recognize       <br>
              ```python
/generate_for_finetune/generate_seq_for_classification_1x.py --data_dir <input_dataset> --output_dir <output_datset> --vocab_file /generate_for_finetune/vocab_v2.txt --max_seq_length 2048
              ```  <br>  <br>
• Finetune:  You just need to feed the formatted data into the MP-Bert model. After processing by MP-BERT, the [CLS] feature is passed into a fully connected network for the final prediction.  <br> 
              ``` python
/finetune_network/mpbert_classification.py --config_path /finetune_network/config_2048.yaml --do_train True --do_eval True --description correlation --num_class 2 --epoch_num <num> --frozen_bert False --device_id <device_target> --data_url <input_dataset> --load_checkpoint_url <MP-BERT_models> --output_url <saved_models> --task_name test --train_batch_size 32 
              ```    <br>  <br>

## 3、Usage
• only need to use the trained model to predict your own target data   <br>
              ``` python
/finetune_network/mpbert_classification.py --config_path /finetune_network/config_2048.yaml --do_predict True --data_url <predicted_data> --description correlation --load_checkpoint_url <your_models> --output_url <output_pathway> --vocab_file /finetune_network/vocab_v2.txt --return_csv True --device_id <device_target>
              ```      <br>  <br> 
## 4、Dataset and models
• Trained models are placed in the 'MPCutter_models' directory
• Datasets required for training the models in the 'MPCutter_cleavage_sites_dataset' directory
• Detailed results of the comparison with different tools in the 'Independent test' directory.
