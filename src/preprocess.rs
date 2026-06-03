use candle::{DType, Device, IndexOp, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RgbFrameShape {
    pub batch: usize,
    pub time: usize,
    pub height: usize,
    pub width: usize,
}

impl RgbFrameShape {
    fn validate(&self) -> Result<()> {
        if self.batch == 0 || self.time == 0 || self.height == 0 || self.width == 0 {
            candle::bail!("RGB frame shape dimensions must all be greater than zero");
        }
        Ok(())
    }

    fn input_len(&self) -> usize {
        self.batch * self.time * self.height * self.width * 3
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePreprocess {
    pub image_size: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImagePreprocess {
    pub fn imagenet_224() -> Self {
        Self {
            image_size: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    fn validate(&self) -> Result<()> {
        if self.image_size == 0 {
            candle::bail!("image_size must be greater than zero");
        }
        if self.std.iter().any(|std| *std == 0.0) {
            candle::bail!("image normalization std values must be non-zero");
        }
        Ok(())
    }
}

pub fn preprocess_rgb_frames_u8(
    frames: &[u8],
    shape: RgbFrameShape,
    config: ImagePreprocess,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    shape.validate()?;
    config.validate()?;
    if frames.len() != shape.input_len() {
        candle::bail!(
            "RGB frame buffer has {} bytes, expected {} for {:?}",
            frames.len(),
            shape.input_len(),
            shape
        );
    }

    let out_size = config.image_size;
    let mut output = vec![0f32; shape.batch * shape.time * 3 * out_size * out_size];
    for b in 0..shape.batch {
        for t in 0..shape.time {
            for y in 0..out_size {
                let src_y = y * shape.height / out_size;
                for x in 0..out_size {
                    let src_x = x * shape.width / out_size;
                    let input_base = ((((b * shape.time + t) * shape.height + src_y) * shape.width
                        + src_x)
                        * 3) as usize;
                    for channel in 0..3 {
                        let output_index = ((((b * shape.time + t) * 3 + channel) * out_size + y)
                            * out_size
                            + x) as usize;
                        let value = frames[input_base + channel] as f32 / 255.0;
                        output[output_index] = (value - config.mean[channel]) / config.std[channel];
                    }
                }
            }
        }
    }

    Tensor::from_vec(
        output,
        (shape.batch, shape.time, 3, out_size, out_size),
        device,
    )?
    .to_dtype(dtype)
}

pub fn preprocess_latest_rgb_frame_u8(
    frames: &[u8],
    shape: RgbFrameShape,
    config: ImagePreprocess,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let time = shape.time;
    let frames = preprocess_rgb_frames_u8(frames, shape, config, dtype, device)?;
    frames.i((.., time - 1, .., .., ..))
}

pub fn preprocess_actions(
    actions: &[f32],
    batch: usize,
    time: usize,
    action_dim: usize,
    min: &[f32],
    max: &[f32],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    if batch == 0 || time == 0 || action_dim == 0 {
        candle::bail!("action shape dimensions must all be greater than zero");
    }
    if actions.len() != batch * time * action_dim {
        candle::bail!(
            "action buffer has {} values, expected {}",
            actions.len(),
            batch * time * action_dim
        );
    }
    if min.len() != action_dim || max.len() != action_dim {
        candle::bail!(
            "action bounds must match action_dim {action_dim}, got min={} max={}",
            min.len(),
            max.len()
        );
    }

    let mut output = Vec::with_capacity(actions.len());
    for chunk in actions.chunks_exact(action_dim) {
        for (idx, value) in chunk.iter().enumerate() {
            output.push(value.clamp(min[idx], max[idx]));
        }
    }

    Tensor::from_vec(output, (batch, time, action_dim), device)?.to_dtype(dtype)
}
