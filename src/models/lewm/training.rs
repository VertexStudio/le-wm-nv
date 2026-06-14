use candle::{IndexOp, Result, Tensor};

use super::{
    LeWm,
    loss::{SigRegConfig, sigreg_loss},
};

#[derive(Debug)]
pub struct LeWmBatchLoss {
    pub total_loss: Tensor,
    pub prediction_loss: Tensor,
    pub sigreg_loss: Tensor,
}

pub fn batch_loss(model: &LeWm, pixels: &Tensor, actions: &Tensor) -> Result<LeWmBatchLoss> {
    batch_loss_with_sigreg(model, pixels, actions, SigRegConfig::default())
}

pub(crate) fn batch_loss_with_sigreg(
    model: &LeWm,
    pixels: &Tensor,
    actions: &Tensor,
    sigreg: SigRegConfig,
) -> Result<LeWmBatchLoss> {
    sigreg.validate()?;
    let pixel_dims = pixels.dims();
    let action_dims = actions.dims();
    if pixel_dims.len() != 5 {
        candle::bail!(
            "LeWM training pixels expect [batch, time, channels, height, width], got {:?}",
            pixels.shape()
        );
    }
    if action_dims.len() != 3 {
        candle::bail!(
            "LeWM training actions expect [batch, time, action_dim], got {:?}",
            actions.shape()
        );
    }
    if pixel_dims[0] != action_dims[0] || pixel_dims[1] != action_dims[1] {
        candle::bail!(
            "LeWM training pixels/actions batch-time mismatch: {:?} vs {:?}",
            pixels.shape(),
            actions.shape()
        );
    }
    if pixel_dims[1] < 2 {
        candle::bail!("LeWM training batch loss requires at least two frames");
    }

    let emb = model.encode_pixels(pixels)?;
    let pred = model.predict(&emb, actions)?;
    let time = emb.dim(1)?;
    let pred_next = pred.i((.., 0..(time - 1), ..))?;
    let target_next = emb.detach().i((.., 1..time, ..))?;
    let prediction_loss = mse_loss(&pred_next, &target_next)?;

    let sigreg_loss = sigreg_loss(&emb.transpose(0, 1)?, sigreg)?;
    let total_loss = (prediction_loss.clone() + (sigreg_loss.clone() * sigreg.weight)?)?;

    Ok(LeWmBatchLoss {
        total_loss,
        prediction_loss,
        sigreg_loss,
    })
}

fn mse_loss(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    (lhs - rhs)?.sqr()?.mean_all()
}
