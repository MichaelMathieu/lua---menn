lua---menn
==========

Memory Efficient Neural Networks for Torch7

Usage:
======

Create and use as a nn.Sequential :
```
local net = menn.Sequential()
net:add(nn.Linear(42, 42))
net:add(nn.Threshold())
net:add(nn.Linear(42, 12))
net:add(nn.LogSoftMax())
```

Once it is fully created (and not before!), you can use the functions inferenceMode
and trainingMode to switch back and forth :
```
local input = torch.Tensor(42)

net:inferenceMode() -- Now you can only use forward
local output = net:forward(input)

net:trainingMode() -- Now you can use backward
local output = net:forward(input) -- You must call forward again, even if you called it before
local gradOutput = ...
local gradInput = net:backward(input, gradOutput)
```

Notes:
======

* For now, do NOT use ParallelTable modules (and try not to use Parallel either, it is not tested.