use tch::{Device, IndexOp, Kind, Tensor};

pub fn random_batch(
    ins: &Tensor,
    outs: &Tensor,
    batch_size: usize,
    device: Device,
) -> (Tensor, Tensor) {
    let _no_grad = tch::no_grad_guard();
    let indices = Tensor::randint(ins.size()[0], &[batch_size as i64], (Kind::Int64, device));
    let ins_batch = ins.i(&indices);
    let outs_batch = outs.i(&indices);
    (ins_batch, outs_batch)
}
