"""Full-vocabulary forward KL with bounded activation memory."""
import torch
from torch.nn import functional as F


def masked_fkl(student, teacher, valid):
    result = student.new_zeros((), dtype=torch.float32)
    for start in range(0, student.shape[1], 32):
        target = F.log_softmax(teacher[:, start:start+32].float(), -1).detach()
        pred = F.log_softmax(student[:, start:start+32].float(), -1)
        value = (target.exp() * (target - pred)).sum(-1)
        result = result + (value * valid[:, start:start+32]).sum()
    return result


class _MemoryBoundedFKL(torch.autograd.Function):
    """Recompute softmax in backward instead of retaining all FP32 probabilities."""
    @staticmethod
    def forward(ctx, student, teacher, valid):
        if student.shape != teacher.shape or student.shape[:2] != valid.shape:
            raise ValueError('KL shapes differ')
        ctx.save_for_backward(student, teacher, valid)
        return masked_fkl(student, teacher, valid)

    @staticmethod
    def backward(ctx, upstream):
        student, teacher, valid = ctx.saved_tensors
        grad = torch.empty_like(student)
        for start in range(0, student.shape[1], 32):
            logp = F.log_softmax(student[:, start:start+32].float(), -1)
            target = F.log_softmax(teacher[:, start:start+32].float(), -1).exp()
            # Use the same native log-softmax VJP and operation order as the
            # reference autograd graph. p-q is algebraically equivalent but its
            # rounding amplified through the deep BF16 model during qualification.
            grad_logp = -(target * (valid[:, start:start+32, None] * upstream))
            value = torch.ops.aten._log_softmax_backward_data(grad_logp, logp, -1, torch.float32)
            grad[:, start:start+32] = value.to(student.dtype)
        return grad, None, None


def memory_bounded_fkl(student, teacher, valid):
    return _MemoryBoundedFKL.apply(student, teacher, valid)
