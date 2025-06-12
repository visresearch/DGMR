import torch

class Knowledge_Distillation_Loss(torch.nn.Module):
    def __init__(self, T = 1):
        super(Knowledge_Distillation_Loss, self).__init__()
        self.KLdiv = torch.nn.KLDivLoss()
        self.T = T

    def forward(self, output_student, output_teacher):
        loss_kl = self.KLdiv(torch.nn.functional.log_softmax(output_student / self.T, dim=1), torch.nn.functional.softmax(output_teacher / self.T, dim=1))

        return loss_kl