import torch
import torch.distributed as dist

class AllToAllTokens(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_split_sizes, in_split_sizes, pg):
        ctx.pg = pg
        ctx.out_split_sizes = out_split_sizes
        ctx.in_split_sizes = in_split_sizes
        y = x.new_empty((sum(out_split_sizes), x.shape[-1]))
        dist.all_to_all_single(y, x, 
                               output_split_sizes=out_split_sizes, 
                               input_split_sizes=in_split_sizes, 
                               group=pg)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        grad_x = grad_y.new_empty((sum(ctx.in_split_sizes), grad_y.shape[-1]))
        dist.all_to_all_single(grad_x, grad_y,
                               output_split_sizes=ctx.in_split_sizes,
                               input_split_sizes=ctx.out_split_sizes,
                               group=ctx.pg)
        return grad_x, None, None, None
