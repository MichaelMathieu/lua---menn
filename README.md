lua---menn
==========

Memory Efficient Neural Networks for Torch7

Usage:
------

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

Tested compatible modules:
--------------------------

On CPU:

* nn.Linear
* nn.Threshold
* nn.LogSoftMax
* nn.SpatialConvolution
* nn.SpatialMaxPooling
* nn.SpatialZeroPadding

Notes:
------

* If, at some point, you use large input or minibatch, and want to use
  smaller ones later, you should call inferenceMode again, or the memory won't
  be freed :
```
net:inferenceMode()

local input_BIG = torch.Tensor(256, 42)
net:forward(input_BIG) -- a lot of memory is used

local input_small = torch.Tensor(4, 42)
net:forward(input_small) -- a lot of memory is *still* used

net:inferenceMode()
net:forward(input_small) -- a small amount of memory is used
```
* For now, do NOT use ParallelTable modules