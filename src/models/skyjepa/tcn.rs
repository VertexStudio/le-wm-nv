use candle::{IndexOp, Module, Result, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder, conv1d};

use super::TemporalConvConfig;

#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: Conv1d,
    left_padding: usize,
}

impl CausalConv1d {
    fn new(
        input_dim: usize,
        output_dim: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let left_padding = dilation
            .checked_mul(kernel_size.saturating_sub(1))
            .ok_or_else(|| candle::Error::Msg("TCN causal padding overflowed".to_string()))?;
        let conv = conv1d(
            input_dim,
            output_dim,
            kernel_size,
            Conv1dConfig {
                dilation,
                ..Conv1dConfig::default()
            },
            vb,
        )?;
        Ok(Self { conv, left_padding })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let padded = if self.left_padding == 0 {
            xs.clone()
        } else {
            xs.pad_with_zeros(2, self.left_padding, 0)?
        };
        self.conv.forward(&padded)
    }
}

#[derive(Debug, Clone)]
struct TemporalBlock {
    conv1: CausalConv1d,
    conv2: CausalConv1d,
    residual: Option<Conv1d>,
}

impl TemporalBlock {
    fn new(
        input_dim: usize,
        output_dim: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 =
            CausalConv1d::new(input_dim, output_dim, kernel_size, dilation, vb.pp("conv1"))?;
        let conv2 = CausalConv1d::new(
            output_dim,
            output_dim,
            kernel_size,
            dilation,
            vb.pp("conv2"),
        )?;
        let residual = if input_dim == output_dim {
            None
        } else {
            Some(conv1d(
                input_dim,
                output_dim,
                1,
                Conv1dConfig::default(),
                vb.pp("residual"),
            )?)
        };
        Ok(Self {
            conv1,
            conv2,
            residual,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let hidden = self.conv1.forward(xs)?.gelu()?;
        let hidden = self.conv2.forward(&hidden)?.gelu()?;
        let residual = match &self.residual {
            Some(layer) => layer.forward(xs)?,
            None => xs.clone(),
        };
        (hidden + residual)?.gelu()
    }
}

/// Compact causal TCN whose input and output remain entirely on the selected
/// Candle device. Inputs use the ergonomic `[batch, time, feature]` layout.
#[derive(Debug, Clone)]
pub struct TemporalConvEncoder {
    input_dim: usize,
    output_dim: usize,
    blocks: Vec<TemporalBlock>,
}

impl TemporalConvEncoder {
    pub fn new(cfg: &TemporalConvConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate("tcn")
            .map_err(|error| candle::Error::Msg(error.to_string()))?;
        let mut blocks = Vec::with_capacity(cfg.channels.len());
        let mut input_dim = cfg.input_dim;
        for (level, &output_dim) in cfg.channels.iter().enumerate() {
            let dilation = 1usize
                .checked_shl(level as u32)
                .ok_or_else(|| candle::Error::Msg("TCN dilation overflowed".to_string()))?;
            blocks.push(TemporalBlock::new(
                input_dim,
                output_dim,
                cfg.kernel_size,
                dilation,
                vb.pp("blocks").pp(level),
            )?);
            input_dim = output_dim;
        }
        Ok(Self {
            input_dim: cfg.input_dim,
            output_dim: cfg.output_dim(),
            blocks,
        })
    }

    pub fn output_dim(&self) -> usize {
        self.output_dim
    }

    pub fn forward_sequence(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, time, input_dim) = xs.dims3()?;
        if input_dim != self.input_dim {
            candle::bail!(
                "TCN input dim {input_dim} does not match configured {}",
                self.input_dim
            );
        }
        if batch == 0 || time == 0 {
            candle::bail!("TCN requires non-empty batch and time dimensions");
        }
        let mut hidden = xs.transpose(1, 2)?.contiguous()?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        hidden.transpose(1, 2)?.contiguous()
    }

    pub fn forward_last(&self, xs: &Tensor) -> Result<Tensor> {
        let sequence = self.forward_sequence(xs)?;
        let time = sequence.dim(1)?;
        sequence.i((.., time - 1, ..))?.contiguous()
    }
}
