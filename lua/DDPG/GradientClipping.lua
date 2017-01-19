local GradientClipping, parent = torch.class('nn.GradientClipping', 'nn.Module')

function GradientClipping:__init(max_grad)
   parent.__init(self)
   self.max_grad = max_grad or 5
end

function GradientClipping:updateOutput(input)
   self.output = input
   return self.output 
end

function GradientClipping:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input)
   self.gradInput:copy(gradOutput)

   self.gradInput:clamp(-self.max_grad, self.max_grad)

   return self.gradInput
end