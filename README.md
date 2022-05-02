# BiteNet

## Data Preprocessing

  Once you have MIMIC III dataset download copy these CSV files ADMISSIONS.CSV, DIAGNOSES_ICD.CSV, DRGCODES.CSV, PRESCRIPTIONS.CSV, PROCEDURES_ICD.CSV  to "dataset/mimic3"
  
  Set "project dir path" in "uitl/config.py" self.project_dir = root_dir + 'path to BiteNet folder'  in config.py

Command: ~<project_path>/dataset$ python3 data_preperation.py

## Training 

### Re-admission Predictions scripts BiteNet_mh_RE.py

sample Command to start traning BiteNet model:

### Training with Dx only (Patient Diagnosis codes)

BiteNet$ python3 train/BiteNet_mh_RE.py --only_dx_flag True --pos_encoding encoding --max_epoch 20 --model Bite --train_batch_size 32 --min_cut_freq 5 --dropout 0.1 --visit_threshold 2


### Training with Dx and Tx (Both Diagnosis and Procedure codes of patient)

BiteNet$ python3 train/BiteNet_mh_RE.py --only_dx_flag False --pos_encoding encoding --max_epoch 20 --model Bite --train_batch_size 32 --min_cut_freq 5 --dropout 0.1 --visit_threshold 2

### Diagnosis Predictions scripts BiteNet_mh_Dx.py

### Training with Dx only (Patient Diagnosis codes)

BiteNet$ python3 train/BiteNet_mh_DX.py --predict_type dx --pos_encoding encoding --max_epoch 20 --model Bite --task BiteNet --train_batch_size 32 --min_cut_freq 5 --dropout 0.1 --visit_threshold 2 --only_dx_flag True

### Training with Dx and Tx (Both Diagnosis and Procedure codes of patient)

BiteNet$ python3 train/BiteNet_mh_DX.py --predict_type dx --pos_encoding encoding --max_epoch 20 --model Bite --task BiteNet --train_batch_size 32 --min_cut_freq 5 --dropout 0.1 --visit_threshold 2 --only_dx_flag False


### Training with fine tunned model parameters

### Diagnosis Predictions with Dx Only:

   Command: ~/<path to project>/BiteNet$ python3 train/BiteNet_mh_Dx.py --predict_type dx --max_epochs 35 --model Bite --dropout 0.1 --embedding_size 50

### Diagnosis Predictions with Dx and Tx:
  
  Command: ~/<path to project>/BiteNet$ python3 train/BiteNet_mh_Dx.py --predict_type dx --max_epochs 35 --model Bite --dropout 0.1 --embedding_size 50 --only_dx_flag False

### Re-admission Predictions with Dx Only:

 Command: ~/<path to project>/BiteNet$ python3 train/BiteNet_mh_RE.py --max_epochs 20 --model Bite --dropout 0.1 --embedding_size 50
  
### Re-admission Predictions with Dx and Tx:

   Command: ~/<path to project>/BiteNet$ python3 train/BiteNet_mh_RE.py --max_epochs 20 --model Bite --dropout 0.1 --embedding_size 100 --only_dx_flag False --num_heads 2

  
  
### Compute
  
  Used Nvidia GTX 1060 GPU with 6GB RAM for training. On an average each epoch of BiteNet model took ~220msec (when trained with Dx alone) and ~320msec (when trained  with both Dx and Tx).
  
