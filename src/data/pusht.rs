use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use candle::{DType, Device, Tensor};
use hdf5::{Dataset, File};
use ndarray::{Array, Ix3, s};
use serde::{Deserialize, Serialize};

use crate::media::{
    ImageHistoryPreprocessor, ImagePreprocess as CudaImagePreprocess, PackedImageFormat,
    PackedImageShape,
};

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PushTBatchConfig {
    pub batch_size: usize,
    pub history_size: usize,
    pub action_block: usize,
    pub image_size: usize,
    pub normalize_actions: bool,
}

impl PushTBatchConfig {
    pub fn validate(self) -> anyhow::Result<()> {
        ensure!(self.batch_size > 0, "batch_size must be greater than zero");
        ensure!(
            self.history_size > 0,
            "history_size must be greater than zero"
        );
        ensure!(
            self.action_block > 0,
            "action_block must be greater than zero"
        );
        ensure!(self.image_size > 0, "image_size must be greater than zero");
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushTActionStats {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PushTDatasetSummary {
    pub path: PathBuf,
    pub rows: usize,
    pub valid_rows: usize,
    pub episodes: usize,
    pub raw_action_dim: usize,
    pub model_action_dim: usize,
    pub pixel_height: usize,
    pub pixel_width: usize,
    pub pixel_channels: usize,
    pub config: PushTBatchConfig,
    pub action_stats: PushTActionStats,
}

#[derive(Debug)]
pub struct PushTBatch {
    pub pixels: Tensor,
    pub actions: Tensor,
    pub meta: PushTBatchMeta,
}

#[derive(Debug, Clone, Serialize)]
pub struct PushTBatchMeta {
    pub rows: Vec<usize>,
    pub episode_idx: Vec<i64>,
    pub step_idx: Vec<i64>,
}

pub struct PushTDataset {
    path: PathBuf,
    config: PushTBatchConfig,
    pixels: Dataset,
    action_values: Vec<f32>,
    episode_idx: Vec<i64>,
    step_idx: Vec<i64>,
    ep_len: Vec<i32>,
    valid_rows: Vec<usize>,
    valid_mask: Vec<bool>,
    action_stats: PushTActionStats,
    rows: usize,
    raw_action_dim: usize,
    pixel_height: usize,
    pixel_width: usize,
    pixel_channels: usize,
}

impl PushTDataset {
    pub fn open(path: impl AsRef<Path>, config: PushTBatchConfig) -> anyhow::Result<Self> {
        config.validate()?;
        let path = path.as_ref();
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let pixels = file
            .dataset("pixels")
            .context("missing PushT HDF5 dataset `pixels`")?;
        let action = file
            .dataset("action")
            .context("missing PushT HDF5 dataset `action`")?;
        let episode_idx_ds = file
            .dataset("episode_idx")
            .context("missing PushT HDF5 dataset `episode_idx`")?;
        let step_idx_ds = file
            .dataset("step_idx")
            .context("missing PushT HDF5 dataset `step_idx`")?;
        let ep_len_ds = file
            .dataset("ep_len")
            .context("missing PushT HDF5 dataset `ep_len`")?;

        let pixel_shape = pixels.shape();
        ensure!(
            pixel_shape.len() == 4,
            "`pixels` must have shape [rows, height, width, channels], got {pixel_shape:?}"
        );
        let rows = pixel_shape[0];
        let pixel_height = pixel_shape[1];
        let pixel_width = pixel_shape[2];
        let pixel_channels = pixel_shape[3];
        ensure!(
            pixel_channels == 3,
            "`pixels` channel dimension must be 3, got {pixel_channels}"
        );

        let action_shape = action.shape();
        ensure!(
            action_shape.len() == 2,
            "`action` must have shape [rows, action_dim], got {action_shape:?}"
        );
        ensure!(
            action_shape[0] == rows,
            "`action` row count {} does not match pixels row count {rows}",
            action_shape[0]
        );
        let raw_action_dim = action_shape[1];
        ensure!(raw_action_dim > 0, "`action` action_dim must be non-zero");

        let episode_idx = read_raw_i64(&episode_idx_ds, "episode_idx")?;
        let step_idx = read_raw_i64(&step_idx_ds, "step_idx")?;
        let ep_len = read_raw_i32(&ep_len_ds, "ep_len")?;
        ensure!(
            episode_idx.len() == rows,
            "`episode_idx` row count {} does not match pixels row count {rows}",
            episode_idx.len()
        );
        ensure!(
            step_idx.len() == rows,
            "`step_idx` row count {} does not match pixels row count {rows}",
            step_idx.len()
        );
        ensure!(!ep_len.is_empty(), "`ep_len` must not be empty");

        let action_values = action
            .read_raw::<f32>()
            .context("failed to read `action` dataset")?;
        ensure!(
            action_values.len() == rows * raw_action_dim,
            "`action` value count {} does not match shape {:?}",
            action_values.len(),
            action_shape
        );

        let action_stats = compute_action_stats(&action_values, raw_action_dim)?;
        let valid_rows = compute_valid_rows(&episode_idx, &step_idx, &ep_len, config)?;
        ensure!(
            !valid_rows.is_empty(),
            "no valid PushT rows for history_size={} action_block={}",
            config.history_size,
            config.action_block
        );
        let mut valid_mask = vec![false; rows];
        for &row in &valid_rows {
            valid_mask[row] = true;
        }

        Ok(Self {
            path: path.to_path_buf(),
            config,
            pixels,
            action_values,
            episode_idx,
            step_idx,
            ep_len,
            valid_rows,
            valid_mask,
            action_stats,
            rows,
            raw_action_dim,
            pixel_height,
            pixel_width,
            pixel_channels,
        })
    }

    pub fn summary(&self) -> PushTDatasetSummary {
        PushTDatasetSummary {
            path: self.path.clone(),
            rows: self.rows,
            valid_rows: self.valid_rows.len(),
            episodes: self.ep_len.len(),
            raw_action_dim: self.raw_action_dim,
            model_action_dim: self.model_action_dim(),
            pixel_height: self.pixel_height,
            pixel_width: self.pixel_width,
            pixel_channels: self.pixel_channels,
            config: self.config,
            action_stats: self.action_stats.clone(),
        }
    }

    pub fn valid_rows(&self) -> &[usize] {
        &self.valid_rows
    }

    pub fn raw_action_dim(&self) -> usize {
        self.raw_action_dim
    }

    pub fn model_action_dim(&self) -> usize {
        self.raw_action_dim * self.config.action_block
    }

    pub fn image_size(&self) -> usize {
        self.config.image_size
    }

    pub fn history_size(&self) -> usize {
        self.config.history_size
    }

    pub fn action_stats(&self) -> &PushTActionStats {
        &self.action_stats
    }

    pub fn shuffled_valid_rows(&self, seed: u64) -> Vec<usize> {
        let mut rows = self.valid_rows.clone();
        shuffle(&mut rows, seed);
        rows
    }

    pub fn batch(
        &self,
        rows: &[usize],
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<PushTBatch> {
        ensure!(!rows.is_empty(), "cannot create an empty PushT batch");
        ensure!(
            rows.len() <= self.config.batch_size,
            "batch has {} rows, configured batch_size is {}",
            rows.len(),
            self.config.batch_size
        );

        let batch = rows.len();
        let time = self.config.history_size;
        let model_action_dim = self.model_action_dim();
        let mut action_values = vec![0f32; batch * time * model_action_dim];
        let mut episode_idx = Vec::with_capacity(batch);
        let mut step_idx = Vec::with_capacity(batch);

        for (batch_idx, &row) in rows.iter().enumerate() {
            self.ensure_valid_row(row)?;
            episode_idx.push(self.episode_idx[row]);
            step_idx.push(self.step_idx[row]);
            for history_idx in 0..time {
                let frame_row = row + history_idx * self.config.action_block;
                self.write_action_history_block(
                    &mut action_values,
                    batch_idx,
                    history_idx,
                    frame_row,
                )?;
            }
        }

        let pixels = self.preprocess_pixel_batch(rows, dtype, device)?;
        let actions = Tensor::from_vec(action_values, (batch, time, model_action_dim), device)?
            .to_dtype(dtype)?;
        Ok(PushTBatch {
            pixels,
            actions,
            meta: PushTBatchMeta {
                rows: rows.to_vec(),
                episode_idx,
                step_idx,
            },
        })
    }

    fn ensure_valid_row(&self, row: usize) -> anyhow::Result<()> {
        ensure!(
            row < self.rows,
            "row {row} is outside dataset rows {}",
            self.rows
        );
        ensure!(
            self.valid_mask[row],
            "row {row} is not valid for history_size={} action_block={}",
            self.config.history_size,
            self.config.action_block
        );
        let last_action_row = row + self.config.history_size * self.config.action_block - 1;
        ensure!(
            last_action_row < self.rows,
            "row {row} would read action row {last_action_row}, beyond dataset rows {}",
            self.rows
        );
        Ok(())
    }

    fn preprocess_pixel_batch(
        &self,
        rows: &[usize],
        dtype: DType,
        device: &Device,
    ) -> anyhow::Result<Tensor> {
        let batch = rows.len();
        let mut preprocessor = ImageHistoryPreprocessor::new(
            device,
            PackedImageShape::new(
                batch,
                self.pixel_height,
                self.pixel_width,
                PackedImageFormat::Rgb,
            ),
            self.config.history_size,
            CudaImagePreprocess {
                output_height: self.config.image_size,
                output_width: self.config.image_size,
                mean: IMAGENET_MEAN,
                std: IMAGENET_STD,
            },
        )?;
        let frame_len = self.pixel_height * self.pixel_width * 3;
        for history_idx in 0..self.config.history_size {
            let mut packed = vec![0u8; batch * frame_len];
            for (batch_idx, &row) in rows.iter().enumerate() {
                let frame_row = row + history_idx * self.config.action_block;
                self.read_pixel_frame_into(&mut packed, batch_idx, frame_row)
                    .with_context(|| {
                        format!("failed to read pixels for row {row} history index {history_idx}")
                    })?;
            }
            let input = Tensor::from_vec(
                packed,
                (batch, self.pixel_height, self.pixel_width, 3),
                device,
            )?;
            preprocessor.preprocess_packed_u8_into_slot(&input, history_idx)?;
        }
        preprocessor
            .output()
            .clone()
            .to_dtype(dtype)
            .map_err(Into::into)
    }

    fn read_pixel_frame_into(
        &self,
        output: &mut [u8],
        batch_idx: usize,
        frame_row: usize,
    ) -> anyhow::Result<()> {
        let image: Array<u8, Ix3> = self
            .pixels
            .read_slice(s![frame_row, .., .., ..])
            .with_context(|| format!("failed to read pixels[{frame_row}]"))?;
        let rgb = image
            .as_slice_memory_order()
            .context("pixels row is not contiguous in memory order")?;
        ensure!(
            rgb.len() == self.pixel_height * self.pixel_width * 3,
            "pixels[{frame_row}] has {} bytes, expected {}",
            rgb.len(),
            self.pixel_height * self.pixel_width * 3
        );
        let frame_len = self.pixel_height * self.pixel_width * 3;
        let start = batch_idx * frame_len;
        let end = start + frame_len;
        ensure!(
            end <= output.len(),
            "pixel output range {start}..{end} exceeds buffer length {}",
            output.len()
        );
        output[start..end].copy_from_slice(rgb);
        Ok(())
    }

    fn write_action_history_block(
        &self,
        output: &mut [f32],
        batch_idx: usize,
        history_idx: usize,
        start_row: usize,
    ) -> anyhow::Result<()> {
        let model_action_dim = self.model_action_dim();
        let base_out = (batch_idx * self.config.history_size + history_idx) * model_action_dim;
        for block_idx in 0..self.config.action_block {
            let row = start_row + block_idx;
            let in_base = row * self.raw_action_dim;
            let out_base = base_out + block_idx * self.raw_action_dim;
            for action_idx in 0..self.raw_action_dim {
                let mut value = self.action_values[in_base + action_idx];
                if self.config.normalize_actions {
                    value = (value - self.action_stats.mean[action_idx])
                        / self.action_stats.std[action_idx];
                }
                output[out_base + action_idx] = value;
            }
        }
        Ok(())
    }
}

fn read_raw_i64(dataset: &Dataset, name: &str) -> anyhow::Result<Vec<i64>> {
    dataset
        .read_raw::<i64>()
        .with_context(|| format!("failed to read `{name}` dataset"))
}

fn read_raw_i32(dataset: &Dataset, name: &str) -> anyhow::Result<Vec<i32>> {
    dataset
        .read_raw::<i32>()
        .with_context(|| format!("failed to read `{name}` dataset"))
}

fn compute_action_stats(values: &[f32], action_dim: usize) -> anyhow::Result<PushTActionStats> {
    ensure!(action_dim > 0, "action_dim must be greater than zero");
    ensure!(
        values.len().is_multiple_of(action_dim),
        "action value count {} is not divisible by action_dim {action_dim}",
        values.len()
    );
    let count = values.len() / action_dim;
    ensure!(count > 0, "action dataset must contain at least one row");
    let mut total = vec![0f64; action_dim];
    let mut total_sq = vec![0f64; action_dim];
    for row in values.chunks_exact(action_dim) {
        for (idx, value) in row.iter().enumerate() {
            let value = f64::from(*value);
            total[idx] += value;
            total_sq[idx] += value * value;
        }
    }
    let mut mean = Vec::with_capacity(action_dim);
    let mut std = Vec::with_capacity(action_dim);
    let count = count as f64;
    for idx in 0..action_dim {
        let m = total[idx] / count;
        let variance = (total_sq[idx] / count - m * m).max(1e-12);
        mean.push(m as f32);
        std.push(variance.sqrt() as f32);
    }
    Ok(PushTActionStats { mean, std })
}

fn compute_valid_rows(
    episode_idx: &[i64],
    step_idx: &[i64],
    ep_len: &[i32],
    config: PushTBatchConfig,
) -> anyhow::Result<Vec<usize>> {
    ensure!(
        episode_idx.len() == step_idx.len(),
        "episode_idx and step_idx lengths differ: {} vs {}",
        episode_idx.len(),
        step_idx.len()
    );
    let history_span = config
        .history_size
        .checked_mul(config.action_block)
        .context("history_size * action_block overflowed")?;
    let mut rows = Vec::new();
    for row in 0..episode_idx.len() {
        let episode = usize::try_from(episode_idx[row])
            .with_context(|| format!("episode_idx[{row}] is negative"))?;
        ensure!(
            episode < ep_len.len(),
            "episode_idx[{row}]={episode} is outside ep_len length {}",
            ep_len.len()
        );
        let valid_until = i64::from(ep_len[episode]) - history_span as i64;
        if step_idx[row] <= valid_until {
            rows.push(row);
        }
    }
    Ok(rows)
}

#[cfg(test)]
#[derive(Debug, Clone, Copy)]
struct PixelOutputLayout {
    batch_idx: usize,
    history_idx: usize,
    batch: usize,
    time: usize,
}

#[cfg(test)]
fn write_normalized_rgb_chw(
    input: &[u8],
    src_h: usize,
    src_w: usize,
    dst_size: usize,
    output: &mut [f32],
    layout: PixelOutputLayout,
) -> anyhow::Result<()> {
    ensure!(
        input.len() == src_h * src_w * 3,
        "RGB input has {} bytes, expected {}",
        input.len(),
        src_h * src_w * 3
    );
    ensure!(
        layout.batch_idx < layout.batch,
        "batch_idx {} must be less than batch {}",
        layout.batch_idx,
        layout.batch
    );
    ensure!(
        layout.history_idx < layout.time,
        "history_idx {} must be less than time {}",
        layout.history_idx,
        layout.time
    );
    let expected = layout.batch * layout.time * 3 * dst_size * dst_size;
    ensure!(
        output.len() == expected,
        "pixel output has {} values, expected {expected}",
        output.len()
    );

    for y in 0..dst_size {
        for x in 0..dst_size {
            let rgb = if src_h == dst_size && src_w == dst_size {
                let base = (y * src_w + x) * 3;
                [
                    input[base] as f32,
                    input[base + 1] as f32,
                    input[base + 2] as f32,
                ]
            } else {
                bilinear_rgb(input, src_h, src_w, y, x, dst_size)
            };
            for channel in 0..3 {
                let value = rgb[channel] / 255.0;
                let out = ((((layout.batch_idx * layout.time + layout.history_idx) * 3 + channel)
                    * dst_size
                    + y)
                    * dst_size)
                    + x;
                output[out] = (value - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
fn bilinear_rgb(
    input: &[u8],
    src_h: usize,
    src_w: usize,
    dst_y: usize,
    dst_x: usize,
    dst_size: usize,
) -> [f32; 3] {
    let src_y = resize_coord(dst_y, dst_size, src_h);
    let src_x = resize_coord(dst_x, dst_size, src_w);
    let y0 = src_y.floor().max(0.0) as usize;
    let x0 = src_x.floor().max(0.0) as usize;
    let y1 = (y0 + 1).min(src_h - 1);
    let x1 = (x0 + 1).min(src_w - 1);
    let wy = src_y - y0 as f32;
    let wx = src_x - x0 as f32;
    let mut rgb = [0f32; 3];
    for channel in 0..3 {
        let p00 = input[(y0 * src_w + x0) * 3 + channel] as f32;
        let p01 = input[(y0 * src_w + x1) * 3 + channel] as f32;
        let p10 = input[(y1 * src_w + x0) * 3 + channel] as f32;
        let p11 = input[(y1 * src_w + x1) * 3 + channel] as f32;
        let top = p00 * (1.0 - wx) + p01 * wx;
        let bottom = p10 * (1.0 - wx) + p11 * wx;
        rgb[channel] = top * (1.0 - wy) + bottom * wy;
    }
    rgb
}

#[cfg(test)]
fn resize_coord(dst_idx: usize, dst_len: usize, src_len: usize) -> f32 {
    if dst_len == 1 {
        0.0
    } else {
        ((dst_idx as f32 + 0.5) * src_len as f32 / dst_len as f32 - 0.5)
            .clamp(0.0, (src_len - 1) as f32)
    }
}

fn shuffle<T>(values: &mut [T], seed: u64) {
    let mut rng = SplitMix64::new(seed);
    for idx in (1..values.len()).rev() {
        let swap_idx = rng.next_usize(idx + 1);
        values.swap(idx, swap_idx);
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn next_usize(&mut self, upper: usize) -> usize {
        (self.next_u64() as usize) % upper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computes_action_stats_per_raw_action_dimension() {
        let stats = compute_action_stats(&[1.0, 2.0, 3.0, 6.0], 2).unwrap();
        assert_eq!(stats.mean, vec![2.0, 4.0]);
        assert!((stats.std[0] - 1.0).abs() < 1e-6);
        assert!((stats.std[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn computes_valid_rows_like_python_exporter() {
        let cfg = PushTBatchConfig {
            batch_size: 2,
            history_size: 3,
            action_block: 2,
            image_size: 224,
            normalize_actions: true,
        };
        let episode_idx = [0, 0, 0, 0, 0, 0, 1, 1, 1];
        let step_idx = [0, 1, 2, 3, 4, 5, 0, 1, 2];
        let ep_len = [8, 4];
        let rows = compute_valid_rows(&episode_idx, &step_idx, &ep_len, cfg).unwrap();
        assert_eq!(rows, vec![0, 1, 2]);
    }

    #[test]
    fn preprocesses_rgb_to_batched_chw_with_imagenet_norm() {
        let mut out = vec![0f32; 1 * 1 * 3];
        write_normalized_rgb_chw(
            &[255, 0, 127],
            1,
            1,
            1,
            &mut out,
            PixelOutputLayout {
                batch_idx: 0,
                history_idx: 0,
                batch: 1,
                time: 1,
            },
        )
        .unwrap();
        assert!((out[0] - ((1.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0])).abs() < 1e-6);
        assert!((out[1] - ((0.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1])).abs() < 1e-6);
        assert!((out[2] - (((127.0 / 255.0) - IMAGENET_MEAN[2]) / IMAGENET_STD[2])).abs() < 1e-6);
    }
}
