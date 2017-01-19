local GradientNormalization, parent = torch.class('nn.GradientNormalization', 'nn.Module')

function GradientNormalization:__init(max_grad_norm)
   parent.__init(self)
   self.max_grad_norm = max_grad_norm or 5
end

function GradientNormalization:updateOutput(input)
   self.output = input
   return self.output 
end

function GradientNormalization:updateGradInput(input, gradOutput)
   local norm = gradOutput:norm()
   if norm < self.max_grad_norm then
      self.gradInput = gradOutput
   else
      self.gradInput:resizeAs(input):copy(gradOutput)
      self.gradInput:mul(self.max_grad_norm/norm)
   end
   return self.gradInput
end