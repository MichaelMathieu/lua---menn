require 'nn'
local Sequential, parent = torch.class('menn.Sequential', 'nn.Module')

function Sequential:__init()
   self.outputs_tmp = {torch.Tensor(), torch.Tensor()}
   self.mode = "training"
   self.modules = {}
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:size()
   return #self.modules
end

function Sequential:get(index)
   return self.modules[index]
end

function Sequential:inferenceMode()
   self.outputs_tmp = {torch.Tensor(), torch.Tensor()}
   for i = 1,#self.modules do
      self.modules[i].output = torch.Tensor()
      self.modules[i].gradInput = torch.Tensor()
   end
   self.mode = "inference"
   collectgarbage()
end

function Sequential:trainingMode()
   self.outputs_tmp = {torch.Tensor(), torch.Tensor()}   
   self.mode = "training"
   collectgarbage()
end

function Sequential:updateOutput(input)
   local currentOutput = input
   if self.mode == "training" then
      for i=1,#self.modules do 
	 currentOutput = self.modules[i]:updateOutput(currentOutput)
      end 
   else
      local tmp_idx = 1
      for i=1,#self.modules do
	 self.modules[i].output = self.outputs_tmp[tmp_idx]
	 currentOutput = self.modules[i]:updateOutput(currentOutput)
	 tmp_idx = 3 - tmp_idx -- alternate, one-based
      end
   end
   self.output = currentOutput
   return currentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   if self.mode ~= "training" then
      error("menn.Sequential : updateGradInput called in inference mode")
   end
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      currentModule = previousModule
   end
   currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   if self.mode ~= "training" then
      error("menn.Sequential : accGradParameters called in inference mode")
   end
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accGradParameters(input, currentGradOutput, scale)
end

function Sequential:accUpdateGradParameters(input, gradOutput, lr)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accUpdateGradParameters(previousModule.output, currentGradOutput, lr)
      currentGradOutput = currentModule.gradInput
      currentModule = previousModule
   end
   
   currentModule:accUpdateGradParameters(input, currentGradOutput, lr)
end

function Sequential:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function Sequential:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function Sequential:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function Sequential:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Sequential:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Sequential:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'menn.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end
