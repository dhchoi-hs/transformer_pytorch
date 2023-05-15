# Training record
- Record of training model that only encoder stack like BERT.

## Training case
### case 1
- description: training model layer 4 and 6
- date: 2023/04/28 20:00~2023/05/02 17:40
- params:
  > batch_size: 256  
learning_rate: 0.0005  
weight_decay: 0
d_model: 512  
h: 8  
ff: 2048  
n_layers: 4 and 6  
p_dropout: 0.1  
seq_len: 256  
- datasets: ko&en unified vocab & dataset
  - trains: 2884352, steps per epoch: 11267 valids: 320484, steps per epoch: 1252
- result: 
  - training loss decreases, valid loss increases. overfitting
  - blue is layer 6, orange is layer 4 in below image
  - (logging iteration with epoch, not step.)
<p align="center"><img src="images/case1_loss.png" width="800"/></p>
<p align="center"><img src="images/case1_acc.png" width="800"/></p>

### case 2
- description: apply L2 weight decay 0.01 to case 1.
- date: 2023/05/04 20:20~2023/05/08 09:30
- params:
  > batch_size: 256  
learning_rate: 0.0005  
weight_decay: 0.01
d_model: 512  
h: 8  
ff: 2048  
n_layers: 4 and 6  
p_dropout: 0.1  
seq_len: 256  
- datasets: ko&en unified vocab & dataset
  - trains: 2884352, steps per epoch: 11267 valids: 320484, steps per epoch: 1252
- result: 
  - underfitting. weight decay is too high.
  - blue is layer 6, orange is layer 4 in below image
  - (logging iteration with epoch, not step.)
<p align="center"><img src="images/case2_loss.png" width="800"/></p>
<p align="center"><img src="images/case2_acc.png" width="800"/></p>

### case 3
- description: 
  - lower weight decay 0.001 to case 2
  - d_model 512 to 768
  - iteration with step.
- date: 2023/05/10 18:00~2023/05/11 10:30
- params:
  > batch_size: 256  
learning_rate: 0.0005  
weight_decay: 0.001  
d_model: 768 and 512  
h: 8  
ff: 2048  
n_layers: 3 and 6  
p_dropout: 0.1  
seq_len: 256  
- datasets: ko&en unified vocab & dataset
  - trains: 2884352, steps per epoch: 11267 valids: 320484, steps per epoch: 1252
- result: 
  - different scores depending on the dataset. english corpus get better score than korean corpus.
  - blue is layer 6, orange is layer 4 in below image
<p align="center"><img src="images/case3_loss.png" width="800"/></p>
<p align="center"><img src="images/case3_acc.png" width="800"/></p>

### case 4.
- description:
  - dataset corpus only ko
  - fixed mask token position on every epoch vs reset mask token position on every epoch
- date: 2023/05/10 18:00~2023/05/11 10:30
- params:
  > batch_size: 256  
learning_rate: 0.0005  
weight_decay: 0.001  
d_model: 768  
h: 8  
ff: 2048  
n_layers: 3  
p_dropout: 0.1  
seq_len: 256  
- datasets: ko&en unified vocab & only ko dataset
  - trains: 2884352, steps per epoch: 11267 valids: 320484, steps per epoch: 1252
- result: 
  - fixed mask token getting better scores slightly than reset mask token.
  - blue is reset mask token, orange is fixed mask token in below image.
<p align="center"><img src="images/case4_loss.png" width="800"/></p>
<p align="center"><img src="images/case4_acc.png" width="800"/></p>

### case 5
- description: 
  - only ko
  - weight decay 0.001 vs 0.0001
- date: 2023/05/12 18:25~2023/05/15 09:25
- params:
  > batch_size: 256  
learning_rate: 0.0005  
weight_decay: 0.001 and 0.0001  
d_model: 768  
h: 8  
ff: 2048  
n_layers: 3  
p_dropout: 0.1  
seq_len: 256  
- datasets: ko&en unified vocab & only ko dataset
  - trains: 1442176, steps per epoch: 5634 valids: 160242, steps per epoch: 626
- result: 
  - weight decay 0.0001 is better.
  - blue is weight decay 0.0001, orange is fweight dacy 0.001 in below image.
<p align="center"><img src="images/case5_loss.png" width="800"/></p>
<p align="center"><img src="images/case5_acc.png" width="800"/></p>

### case 6
 - description:
   - only en corpus, size of BPE vocab to 30000, layer 3 vs 6
- date: ...