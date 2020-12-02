# Sequential MNIST
## Usage

```
python sMNIST_task.py [args]
```

Options:
- nhid : hidden size of recurrent net
- epochs : max epochs
- batch : batch size
- lr : learning rate
- dt : step size parameter dt of the coRNN
- gamma : y controle parameter gamma of the coRNN
- epsilon : z controle parameter epsilon of the coRNN

The log of the run with a fixed random seed can be found in the results directory.
Note that the experiment was conducted on an Intel Xeon E3-1585Lv5 CPU. 
Changing the device might result in a change of the random generator.
