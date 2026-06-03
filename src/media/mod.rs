//! NVIDIA/CUDA media ingestion primitives.
//!
//! This module is intentionally CUDA-native: it owns reusable Candle tensors and
//! launches preprocessing kernels on the same CUDA stream Candle uses for model
//! execution.

pub mod nvdec;
pub mod nvjpeg;

use std::{ffi::c_void, fmt, sync::OnceLock};

use candle::{
    CudaStorage, DType, Device, InplaceOp2, InplaceOp3, Layout, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda::{
        WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
    op::BackpropOp,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedImageShape {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub format: PackedImageFormat,
}

impl PackedImageShape {
    pub fn new(batch: usize, height: usize, width: usize, format: PackedImageFormat) -> Self {
        Self {
            batch,
            height,
            width,
            format,
        }
    }

    pub fn channels(self) -> usize {
        self.format.channels()
    }

    pub fn elem_count(self) -> usize {
        self.batch * self.height * self.width * self.channels()
    }

    fn validate(self) -> Result<()> {
        if self.batch == 0 || self.height == 0 || self.width == 0 {
            candle::bail!("packed CUDA image dimensions must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedImageFormat {
    Rgb,
    Bgr,
    Rgba,
    Bgra,
}

impl PackedImageFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Rgb => 0,
            Self::Bgr => 1,
            Self::Rgba => 2,
            Self::Bgra => 3,
        }
    }
}

impl TryFrom<u32> for PackedImageFormat {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Rgb),
            1 => Ok(Self::Bgr),
            2 => Ok(Self::Rgba),
            3 => Ok(Self::Bgra),
            other => candle::bail!(
                "unknown packed CUDA image format {other}; expected 0=RGB, 1=BGR, 2=RGBA, or 3=BGRA"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Nv12ImageShape {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
}

impl Nv12ImageShape {
    pub fn new(batch: usize, height: usize, width: usize) -> Self {
        Self {
            batch,
            height,
            width,
        }
    }

    fn validate(self) -> Result<()> {
        if self.batch == 0 || self.height == 0 || self.width == 0 {
            candle::bail!("NV12 CUDA image dimensions must be greater than zero");
        }
        if self.height % 2 != 0 || self.width % 2 != 0 {
            candle::bail!(
                "NV12 CUDA image dimensions must be even, got {}x{}",
                self.height,
                self.width
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Nv12ColorSpace {
    Bt601Video,
    Bt709Video,
    Bt601Full,
    Bt709Full,
}

impl Nv12ColorSpace {
    fn kernel_id(self) -> u32 {
        match self {
            Self::Bt601Video => 0,
            Self::Bt709Video => 1,
            Self::Bt601Full => 2,
            Self::Bt709Full => 3,
        }
    }
}

impl TryFrom<u32> for Nv12ColorSpace {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Bt601Video),
            1 => Ok(Self::Bt709Video),
            2 => Ok(Self::Bt601Full),
            3 => Ok(Self::Bt709Full),
            other => candle::bail!(
                "unknown NV12 color space {other}; expected 0=BT.601 video, 1=BT.709 video, 2=BT.601 full, or 3=BT.709 full"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePreprocess {
    pub output_height: usize,
    pub output_width: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ImagePreprocess {
    pub fn imagenet_224() -> Self {
        Self {
            output_height: 224,
            output_width: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    fn validate(self) -> Result<()> {
        if self.output_height == 0 || self.output_width == 0 {
            candle::bail!("CUDA preprocess output dimensions must be greater than zero");
        }
        if self.std.iter().any(|value| *value == 0.0) {
            candle::bail!("CUDA preprocess std values must be non-zero");
        }
        Ok(())
    }
}

pub struct Nv12Preprocessor {
    input_shape: Nv12ImageShape,
    color_space: Nv12ColorSpace,
    config: ImagePreprocess,
    output: Tensor,
}

impl fmt::Debug for Nv12Preprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nv12Preprocessor")
            .field("input_shape", &self.input_shape)
            .field("color_space", &self.color_space)
            .field("config", &self.config)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl Nv12Preprocessor {
    pub fn new(
        device: &Device,
        input_shape: Nv12ImageShape,
        color_space: Nv12ColorSpace,
        config: ImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if !device.is_cuda() {
            candle::bail!("Nv12Preprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            color_space,
            config,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn input_shape(&self) -> Nv12ImageShape {
        self.input_shape
    }

    pub fn config(&self) -> ImagePreprocess {
        self.config
    }

    pub fn color_space(&self) -> Nv12ColorSpace {
        self.color_space
    }

    pub fn preprocess_nv12(&mut self, y_plane: &Tensor, uv_plane: &Tensor) -> Result<&Tensor> {
        validate_nv12_tensors(y_plane, uv_plane, self.input_shape)?;
        let op = Nv12ToNchwF32 {
            input_shape: self.input_shape,
            color_space: self.color_space,
            config: self.config,
            output_layout: MediaOutputLayout::Latest,
        };
        self.output.inplace_op3(y_plane, uv_plane, &op)?;
        Ok(&self.output)
    }
}

pub struct ImagePreprocessor {
    input_shape: PackedImageShape,
    config: ImagePreprocess,
    output: Tensor,
}

impl fmt::Debug for ImagePreprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImagePreprocessor")
            .field("input_shape", &self.input_shape)
            .field("config", &self.config)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl ImagePreprocessor {
    pub fn new(
        device: &Device,
        input_shape: PackedImageShape,
        config: ImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if !device.is_cuda() {
            candle::bail!("ImagePreprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            config,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn input_shape(&self) -> PackedImageShape {
        self.input_shape
    }

    pub fn config(&self) -> ImagePreprocess {
        self.config
    }

    pub fn preprocess_packed_u8(&mut self, input: &Tensor) -> Result<&Tensor> {
        validate_input_tensor(input, self.input_shape)?;
        let op = PackedU8ToNchwF32 {
            input_shape: self.input_shape,
            config: self.config,
            output_layout: MediaOutputLayout::Latest,
        };
        self.output.inplace_op2(input, &op)?;
        Ok(&self.output)
    }
}

pub struct Nv12HistoryPreprocessor {
    input_shape: Nv12ImageShape,
    color_space: Nv12ColorSpace,
    config: ImagePreprocess,
    history_len: usize,
    output: Tensor,
}

impl fmt::Debug for Nv12HistoryPreprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Nv12HistoryPreprocessor")
            .field("input_shape", &self.input_shape)
            .field("color_space", &self.color_space)
            .field("config", &self.config)
            .field("history_len", &self.history_len)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl Nv12HistoryPreprocessor {
    pub fn new(
        device: &Device,
        input_shape: Nv12ImageShape,
        color_space: Nv12ColorSpace,
        history_len: usize,
        config: ImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if history_len == 0 {
            candle::bail!("CUDA NV12 history length must be greater than zero");
        }
        if !device.is_cuda() {
            candle::bail!("Nv12HistoryPreprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                history_len,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            color_space,
            config,
            history_len,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn input_shape(&self) -> Nv12ImageShape {
        self.input_shape
    }

    pub fn history_len(&self) -> usize {
        self.history_len
    }

    pub fn config(&self) -> ImagePreprocess {
        self.config
    }

    pub fn color_space(&self) -> Nv12ColorSpace {
        self.color_space
    }

    pub fn preprocess_nv12_into_slot(
        &mut self,
        y_plane: &Tensor,
        uv_plane: &Tensor,
        history_slot: usize,
    ) -> Result<&Tensor> {
        validate_nv12_tensors(y_plane, uv_plane, self.input_shape)?;
        if history_slot >= self.history_len {
            candle::bail!(
                "CUDA NV12 history slot {history_slot} is outside history_len {}",
                self.history_len
            );
        }
        let op = Nv12ToNchwF32 {
            input_shape: self.input_shape,
            color_space: self.color_space,
            config: self.config,
            output_layout: MediaOutputLayout::History {
                history_len: self.history_len,
                history_slot,
            },
        };
        self.output.inplace_op3(y_plane, uv_plane, &op)?;
        Ok(&self.output)
    }
}

pub struct ImageHistoryPreprocessor {
    input_shape: PackedImageShape,
    config: ImagePreprocess,
    history_len: usize,
    output: Tensor,
}

impl fmt::Debug for ImageHistoryPreprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageHistoryPreprocessor")
            .field("input_shape", &self.input_shape)
            .field("config", &self.config)
            .field("history_len", &self.history_len)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl ImageHistoryPreprocessor {
    pub fn new(
        device: &Device,
        input_shape: PackedImageShape,
        history_len: usize,
        config: ImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if history_len == 0 {
            candle::bail!("CUDA image history length must be greater than zero");
        }
        if !device.is_cuda() {
            candle::bail!("ImageHistoryPreprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                history_len,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            config,
            history_len,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn input_shape(&self) -> PackedImageShape {
        self.input_shape
    }

    pub fn history_len(&self) -> usize {
        self.history_len
    }

    pub fn config(&self) -> ImagePreprocess {
        self.config
    }

    pub fn preprocess_packed_u8_into_slot(
        &mut self,
        input: &Tensor,
        history_slot: usize,
    ) -> Result<&Tensor> {
        validate_input_tensor(input, self.input_shape)?;
        if history_slot >= self.history_len {
            candle::bail!(
                "CUDA image history slot {history_slot} is outside history_len {}",
                self.history_len
            );
        }
        let op = PackedU8ToNchwF32 {
            input_shape: self.input_shape,
            config: self.config,
            output_layout: MediaOutputLayout::History {
                history_len: self.history_len,
                history_slot,
            },
        };
        self.output.inplace_op2(input, &op)?;
        Ok(&self.output)
    }
}

fn validate_input_tensor(input: &Tensor, shape: PackedImageShape) -> Result<()> {
    if !input.device().is_cuda() {
        candle::bail!("packed image input must live on a CUDA Candle device");
    }
    if input.dtype() != DType::U8 {
        candle::bail!(
            "packed image input must use U8 dtype, got {:?}",
            input.dtype()
        );
    }
    let expected = [shape.batch, shape.height, shape.width, shape.channels()];
    if input.dims() != expected {
        candle::bail!(
            "packed image input shape mismatch: expected {:?}, got {:?}",
            expected,
            input.dims()
        );
    }
    Ok(())
}

fn validate_nv12_tensors(y_plane: &Tensor, uv_plane: &Tensor, shape: Nv12ImageShape) -> Result<()> {
    if !y_plane.device().is_cuda() || !uv_plane.device().is_cuda() {
        candle::bail!("NV12 input tensors must live on a CUDA Candle device");
    }
    if y_plane.dtype() != DType::U8 || uv_plane.dtype() != DType::U8 {
        candle::bail!(
            "NV12 input tensors must use U8 dtype, got y={:?}, uv={:?}",
            y_plane.dtype(),
            uv_plane.dtype()
        );
    }
    let expected_y = [shape.batch, shape.height, shape.width];
    if y_plane.dims() != expected_y {
        candle::bail!(
            "NV12 Y plane shape mismatch: expected {:?}, got {:?}",
            expected_y,
            y_plane.dims()
        );
    }
    let expected_uv = [shape.batch, shape.height / 2, shape.width / 2, 2];
    if uv_plane.dims() != expected_uv {
        candle::bail!(
            "NV12 UV plane shape mismatch: expected {:?}, got {:?}",
            expected_uv,
            uv_plane.dims()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PackedU8ToNchwF32 {
    input_shape: PackedImageShape,
    config: ImagePreprocess,
    output_layout: MediaOutputLayout,
}

#[derive(Debug, Clone, Copy)]
enum MediaOutputLayout {
    Latest,
    History {
        history_len: usize,
        history_slot: usize,
    },
}

#[derive(Debug, Clone, Copy)]
struct Nv12ToNchwF32 {
    input_shape: Nv12ImageShape,
    color_space: Nv12ColorSpace,
    config: ImagePreprocess,
    output_layout: MediaOutputLayout,
}

impl InplaceOp2 for PackedU8ToNchwF32 {
    fn name(&self) -> &'static str {
        "packed-u8-to-nchw-f32"
    }

    fn cpu_fwd(
        &self,
        _output: &mut candle::CpuStorage,
        _output_layout: &Layout,
        _input: &candle::CpuStorage,
        _input_layout: &Layout,
    ) -> Result<()> {
        candle::bail!("packed-u8-to-nchw-f32 is a CUDA media operation")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        input: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<()> {
        validate_output_layout(
            output,
            output_layout,
            self.input_shape,
            self.config,
            self.output_layout,
        )?;
        validate_input_layout(input, input_layout, self.input_shape)?;

        let cuda = output.device.clone();
        let input = input.as_cuda_slice::<u8>()?;
        let mut output = output.as_cuda_slice_mut::<f32>()?;
        let input = contiguous_slice(input, input_layout, "packed image input")?;
        let mut output = contiguous_slice_mut(&mut output, output_layout, "model image output")?;

        let ptx = cached_media_ptx(
            &PACKED_U8_TO_NCHW_F32_PTX,
            PACKED_U8_TO_NCHW_F32_CUDA,
            "packed-u8-to-nchw-f32",
        )?;
        let func =
            cuda.get_or_load_custom_func("swm_packed_u8_to_nchw_f32", "swm_media_preprocess", ptx)?;

        let elem_count =
            self.input_shape.batch * 3 * self.config.output_height * self.config.output_width;
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let elem_count_u32 = elem_count as u32;
        let batch = self.input_shape.batch as u32;
        let in_h = self.input_shape.height as u32;
        let in_w = self.input_shape.width as u32;
        let channels = self.input_shape.channels() as u32;
        let out_h = self.config.output_height as u32;
        let out_w = self.config.output_width as u32;
        let format = self.input_shape.format.kernel_id();
        let (history_len, history_slot) = match self.output_layout {
            MediaOutputLayout::Latest => (0u32, 0u32),
            MediaOutputLayout::History {
                history_len,
                history_slot,
            } => (history_len as u32, history_slot as u32),
        };
        let mut builder = func.builder();
        builder.arg(&input);
        builder.arg(&mut output);
        builder.arg(&elem_count_u32);
        builder.arg(&batch);
        builder.arg(&in_h);
        builder.arg(&in_w);
        builder.arg(&channels);
        builder.arg(&out_h);
        builder.arg(&out_w);
        builder.arg(&format);
        builder.arg(&history_len);
        builder.arg(&history_slot);
        builder.arg(&self.config.mean[0]);
        builder.arg(&self.config.mean[1]);
        builder.arg(&self.config.mean[2]);
        builder.arg(&self.config.std[0]);
        builder.arg(&self.config.std[1]);
        builder.arg(&self.config.std[2]);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

impl InplaceOp3 for Nv12ToNchwF32 {
    fn name(&self) -> &'static str {
        "nv12-to-nchw-f32"
    }

    fn cpu_fwd(
        &self,
        _output: &mut candle::CpuStorage,
        _output_layout: &Layout,
        _y_plane: &candle::CpuStorage,
        _y_layout: &Layout,
        _uv_plane: &candle::CpuStorage,
        _uv_layout: &Layout,
    ) -> Result<()> {
        candle::bail!("nv12-to-nchw-f32 is a CUDA media operation")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        y_plane: &CudaStorage,
        y_layout: &Layout,
        uv_plane: &CudaStorage,
        uv_layout: &Layout,
    ) -> Result<()> {
        validate_nv12_output_layout(
            output,
            output_layout,
            self.input_shape,
            self.config,
            self.output_layout,
        )?;
        validate_nv12_y_layout(y_plane, y_layout, self.input_shape)?;
        validate_nv12_uv_layout(uv_plane, uv_layout, self.input_shape)?;

        let cuda = output.device.clone();
        let y_plane = y_plane.as_cuda_slice::<u8>()?;
        let uv_plane = uv_plane.as_cuda_slice::<u8>()?;
        let mut output = output.as_cuda_slice_mut::<f32>()?;
        let y_plane = contiguous_slice(y_plane, y_layout, "NV12 Y plane")?;
        let uv_plane = contiguous_slice(uv_plane, uv_layout, "NV12 UV plane")?;
        let mut output = contiguous_slice_mut(&mut output, output_layout, "model image output")?;

        let ptx = cached_media_ptx(
            &NV12_TO_NCHW_F32_PTX,
            NV12_TO_NCHW_F32_CUDA,
            "nv12-to-nchw-f32",
        )?;
        let func =
            cuda.get_or_load_custom_func("swm_nv12_to_nchw_f32", "swm_cuda_nv12_preprocess", ptx)?;

        let pixel_count =
            self.input_shape.batch * self.config.output_height * self.config.output_width;
        let cfg = LaunchConfig::for_num_elems(pixel_count as u32);
        let pixel_count_u32 = pixel_count as u32;
        let batch = self.input_shape.batch as u32;
        let in_h = self.input_shape.height as u32;
        let in_w = self.input_shape.width as u32;
        let out_h = self.config.output_height as u32;
        let out_w = self.config.output_width as u32;
        let color_space = self.color_space.kernel_id();
        let (history_len, history_slot) = match self.output_layout {
            MediaOutputLayout::Latest => (0u32, 0u32),
            MediaOutputLayout::History {
                history_len,
                history_slot,
            } => (history_len as u32, history_slot as u32),
        };
        let mut builder = func.builder();
        builder.arg(&y_plane);
        builder.arg(&uv_plane);
        builder.arg(&mut output);
        builder.arg(&pixel_count_u32);
        builder.arg(&batch);
        builder.arg(&in_h);
        builder.arg(&in_w);
        builder.arg(&out_h);
        builder.arg(&out_w);
        builder.arg(&color_space);
        builder.arg(&history_len);
        builder.arg(&history_slot);
        builder.arg(&self.config.mean[0]);
        builder.arg(&self.config.mean[1]);
        builder.arg(&self.config.mean[2]);
        builder.arg(&self.config.std[0]);
        builder.arg(&self.config.std[1]);
        builder.arg(&self.config.std[2]);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

fn validate_output_layout(
    output: &CudaStorage,
    layout: &Layout,
    input_shape: PackedImageShape,
    config: ImagePreprocess,
    output_layout: MediaOutputLayout,
) -> Result<()> {
    if output.dtype() != DType::F32 {
        candle::bail!("CUDA media output tensor must use F32 dtype");
    }
    let expected = match output_layout {
        MediaOutputLayout::Latest => vec![
            input_shape.batch,
            3,
            config.output_height,
            config.output_width,
        ],
        MediaOutputLayout::History { history_len, .. } => vec![
            input_shape.batch,
            history_len,
            3,
            config.output_height,
            config.output_width,
        ],
    };
    if layout.dims() != expected {
        candle::bail!(
            "CUDA media output shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA media output")?;
    Ok(())
}

fn validate_nv12_output_layout(
    output: &CudaStorage,
    layout: &Layout,
    input_shape: Nv12ImageShape,
    config: ImagePreprocess,
    output_layout: MediaOutputLayout,
) -> Result<()> {
    if output.dtype() != DType::F32 {
        candle::bail!("CUDA NV12 output tensor must use F32 dtype");
    }
    let expected = match output_layout {
        MediaOutputLayout::Latest => vec![
            input_shape.batch,
            3,
            config.output_height,
            config.output_width,
        ],
        MediaOutputLayout::History { history_len, .. } => vec![
            input_shape.batch,
            history_len,
            3,
            config.output_height,
            config.output_width,
        ],
    };
    if layout.dims() != expected {
        candle::bail!(
            "CUDA NV12 output shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA NV12 output")?;
    Ok(())
}

fn validate_input_layout(
    input: &CudaStorage,
    layout: &Layout,
    input_shape: PackedImageShape,
) -> Result<()> {
    if input.dtype() != DType::U8 {
        candle::bail!("CUDA media input tensor must use U8 dtype");
    }
    let expected = [
        input_shape.batch,
        input_shape.height,
        input_shape.width,
        input_shape.channels(),
    ];
    if layout.dims() != expected {
        candle::bail!(
            "CUDA media input shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA media input")?;
    Ok(())
}

fn validate_nv12_y_layout(
    input: &CudaStorage,
    layout: &Layout,
    input_shape: Nv12ImageShape,
) -> Result<()> {
    if input.dtype() != DType::U8 {
        candle::bail!("CUDA NV12 Y plane must use U8 dtype");
    }
    let expected = [input_shape.batch, input_shape.height, input_shape.width];
    if layout.dims() != expected {
        candle::bail!(
            "CUDA NV12 Y plane shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA NV12 Y plane")?;
    Ok(())
}

fn validate_nv12_uv_layout(
    input: &CudaStorage,
    layout: &Layout,
    input_shape: Nv12ImageShape,
) -> Result<()> {
    if input.dtype() != DType::U8 {
        candle::bail!("CUDA NV12 UV plane must use U8 dtype");
    }
    let expected = [
        input_shape.batch,
        input_shape.height / 2,
        input_shape.width / 2,
        2,
    ];
    if layout.dims() != expected {
        candle::bail!(
            "CUDA NV12 UV plane shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA NV12 UV plane")?;
    Ok(())
}

fn require_contiguous(layout: &Layout, name: &str) -> Result<()> {
    if layout.contiguous_offsets().is_none() {
        candle::bail!("{name} must be contiguous");
    }
    Ok(())
}

fn contiguous_slice<'a, T>(
    slice: &'a candle::cuda::cudarc::driver::CudaSlice<T>,
    layout: &Layout,
    name: &str,
) -> Result<candle::cuda::cudarc::driver::CudaView<'a, T>> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("{name} must be contiguous");
    };
    Ok(slice.slice(start..end))
}

fn contiguous_slice_mut<'a, T>(
    slice: &'a mut candle::cuda::cudarc::driver::CudaSlice<T>,
    layout: &Layout,
    name: &str,
) -> Result<candle::cuda::cudarc::driver::CudaViewMut<'a, T>> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("{name} must be contiguous");
    };
    Ok(slice.slice_mut(start..end))
}

pub fn packed_u8_tensor_from_host(
    bytes: &[u8],
    shape: PackedImageShape,
    device: &Device,
) -> Result<Tensor> {
    shape.validate()?;
    if !device.is_cuda() {
        candle::bail!("packed_u8_tensor_from_host requires a CUDA Candle device");
    }
    if bytes.len() != shape.elem_count() {
        candle::bail!(
            "packed image buffer has {} bytes, expected {}",
            bytes.len(),
            shape.elem_count()
        );
    }
    Tensor::from_slice(
        bytes,
        (shape.batch, shape.height, shape.width, shape.channels()),
        device,
    )
}

pub fn packed_u8_tensor(shape: PackedImageShape, device: &Device) -> Result<Tensor> {
    shape.validate()?;
    if !device.is_cuda() {
        candle::bail!("packed_u8_tensor requires a CUDA Candle device");
    }
    Tensor::zeros(
        (shape.batch, shape.height, shape.width, shape.channels()),
        DType::U8,
        device,
    )
}

pub fn nv12_tensors_from_host(
    y_plane: &[u8],
    uv_plane: &[u8],
    shape: Nv12ImageShape,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    shape.validate()?;
    if !device.is_cuda() {
        candle::bail!("nv12_tensors_from_host requires a CUDA Candle device");
    }
    let expected_y = shape.batch * shape.height * shape.width;
    if y_plane.len() != expected_y {
        candle::bail!(
            "NV12 Y plane has {} bytes, expected {}",
            y_plane.len(),
            expected_y
        );
    }
    let expected_uv = shape.batch * (shape.height / 2) * (shape.width / 2) * 2;
    if uv_plane.len() != expected_uv {
        candle::bail!(
            "NV12 UV plane has {} bytes, expected {}",
            uv_plane.len(),
            expected_uv
        );
    }
    let y = Tensor::from_slice(y_plane, (shape.batch, shape.height, shape.width), device)?;
    let uv = Tensor::from_slice(
        uv_plane,
        (shape.batch, shape.height / 2, shape.width / 2, 2),
        device,
    )?;
    Ok((y, uv))
}

pub fn nv12_tensors(shape: Nv12ImageShape, device: &Device) -> Result<(Tensor, Tensor)> {
    shape.validate()?;
    if !device.is_cuda() {
        candle::bail!("nv12_tensors requires a CUDA Candle device");
    }
    let y = Tensor::zeros((shape.batch, shape.height, shape.width), DType::U8, device)?;
    let uv = Tensor::zeros(
        (shape.batch, shape.height / 2, shape.width / 2, 2),
        DType::U8,
        device,
    )?;
    Ok((y, uv))
}

pub fn cuda_u8_tensor_device_ptr(tensor: &Tensor) -> Result<*mut c_void> {
    if !tensor.device().is_cuda() {
        candle::bail!("CUDA media pointer query requires a CUDA Candle tensor");
    }
    if tensor.dtype() != DType::U8 {
        candle::bail!(
            "CUDA media pointer query requires U8 tensor, got {:?}",
            tensor.dtype()
        );
    }
    let (storage, layout) = tensor.storage_and_layout();
    let Storage::Cuda(storage) = &*storage else {
        candle::bail!("CUDA media pointer query requires CUDA storage");
    };
    let slice = storage.as_cuda_slice::<u8>()?;
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("CUDA media pointer query requires contiguous tensor layout");
    };
    let view = slice.slice(start..end);
    let stream = storage.device.cuda_stream();
    let (ptr, _record) = view.view_ptr(&stream);
    Ok(ptr as usize as *mut c_void)
}

pub fn tensor_from_cuda_slice_f32(
    slice: candle::cuda::cudarc::driver::CudaSlice<f32>,
    shape: impl Into<candle::Shape>,
    device: candle::CudaDevice,
) -> Tensor {
    let storage = CudaStorage::wrap_cuda_slice(slice, device);
    Tensor::from_storage(Storage::Cuda(storage), shape, BackpropOp::none(), false)
}

static PACKED_U8_TO_NCHW_F32_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();
static NV12_TO_NCHW_F32_PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn cached_media_ptx(
    cache: &'static OnceLock<std::result::Result<String, String>>,
    source: &'static str,
    name: &'static str,
) -> Result<&'static str> {
    let cached = cache.get_or_init(|| {
        nvrtc::safe::compile_ptx_with_opts(
            source,
            nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .map(|ptx| ptx.to_src())
        .map_err(|err| err.to_string())
    });
    match cached {
        Ok(ptx) => Ok(ptx.as_str()),
        Err(err) => candle::bail!("{name} NVRTC compile failed: {err}"),
    }
}

const PACKED_U8_TO_NCHW_F32_CUDA: &str = r#"
extern "C" __global__ void swm_packed_u8_to_nchw_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int elem_count,
    unsigned int batch,
    unsigned int in_h,
    unsigned int in_w,
    unsigned int channels,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int format,
    unsigned int history_len,
    unsigned int history_slot,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elem_count) {
        return;
    }

    unsigned int x = idx % out_w;
    unsigned int tmp = idx / out_w;
    unsigned int y = tmp % out_h;
    tmp /= out_h;
    unsigned int c = tmp % 3;
    unsigned int b = tmp / 3;
    if (b >= batch) {
        return;
    }

    float src_y = ((float)y + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;
    float src_x = ((float)x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
    if (src_y < 0.0f) {
        src_y = 0.0f;
    }
    if (src_x < 0.0f) {
        src_x = 0.0f;
    }

    unsigned int y0 = (unsigned int)floorf(src_y);
    unsigned int x0 = (unsigned int)floorf(src_x);
    unsigned int y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    unsigned int x1 = x0 + 1 < in_w ? x0 + 1 : x0;
    float wy = src_y - (float)y0;
    float wx = src_x - (float)x0;

    unsigned int src_c = c;
    if (format == 1u || format == 3u) {
        src_c = 2u - c;
    }

    unsigned int base00 = (((b * in_h + y0) * in_w + x0) * channels + src_c);
    unsigned int base01 = (((b * in_h + y0) * in_w + x1) * channels + src_c);
    unsigned int base10 = (((b * in_h + y1) * in_w + x0) * channels + src_c);
    unsigned int base11 = (((b * in_h + y1) * in_w + x1) * channels + src_c);

    float v00 = (float)input[base00];
    float v01 = (float)input[base01];
    float v10 = (float)input[base10];
    float v11 = (float)input[base11];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    float value = (top + (bottom - top) * wy) * 0.00392156862745098f;

    float mean = c == 0u ? mean0 : (c == 1u ? mean1 : mean2);
    float std = c == 0u ? std0 : (c == 1u ? std1 : std2);
    float normalized = (value - mean) / std;
    if (history_len == 0u) {
        output[idx] = normalized;
    } else {
        unsigned int out_idx =
            ((((b * history_len + history_slot) * 3u + c) * out_h + y) * out_w + x);
        output[out_idx] = normalized;
    }
}
"#;

const NV12_TO_NCHW_F32_CUDA: &str = r#"
__device__ float swm_clamp(float value, float low, float high) {
    return fminf(fmaxf(value, low), high);
}

__device__ float swm_sample_plane(
    const unsigned char* __restrict__ plane,
    unsigned int b,
    unsigned int h,
    unsigned int w,
    unsigned int channels,
    unsigned int channel,
    float src_y,
    float src_x
) {
    src_y = swm_clamp(src_y, 0.0f, (float)(h - 1u));
    src_x = swm_clamp(src_x, 0.0f, (float)(w - 1u));
    unsigned int y0 = (unsigned int)floorf(src_y);
    unsigned int x0 = (unsigned int)floorf(src_x);
    unsigned int y1 = y0 + 1u < h ? y0 + 1u : y0;
    unsigned int x1 = x0 + 1u < w ? x0 + 1u : x0;
    float wy = src_y - (float)y0;
    float wx = src_x - (float)x0;

    unsigned int base00 = (((b * h + y0) * w + x0) * channels + channel);
    unsigned int base01 = (((b * h + y0) * w + x1) * channels + channel);
    unsigned int base10 = (((b * h + y1) * w + x0) * channels + channel);
    unsigned int base11 = (((b * h + y1) * w + x1) * channels + channel);

    float v00 = (float)plane[base00];
    float v01 = (float)plane[base01];
    float v10 = (float)plane[base10];
    float v11 = (float)plane[base11];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    return top + (bottom - top) * wy;
}

extern "C" __global__ void swm_nv12_to_nchw_f32(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    float* __restrict__ output,
    unsigned int pixel_count,
    unsigned int batch,
    unsigned int in_h,
    unsigned int in_w,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int color_space,
    unsigned int history_len,
    unsigned int history_slot,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pixel_count) {
        return;
    }

    unsigned int x = idx % out_w;
    unsigned int tmp = idx / out_w;
    unsigned int y = tmp % out_h;
    unsigned int b = tmp / out_h;
    if (b >= batch) {
        return;
    }

    float src_y = ((float)y + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;
    float src_x = ((float)x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
    float y_value = swm_sample_plane(y_plane, b, in_h, in_w, 1u, 0u, src_y, src_x);
    float u_value = swm_sample_plane(
        uv_plane,
        b,
        in_h / 2u,
        in_w / 2u,
        2u,
        0u,
        src_y * 0.5f,
        src_x * 0.5f
    ) - 128.0f;
    float v_value = swm_sample_plane(
        uv_plane,
        b,
        in_h / 2u,
        in_w / 2u,
        2u,
        1u,
        src_y * 0.5f,
        src_x * 0.5f
    ) - 128.0f;

    float r;
    float g;
    float blue;
    if (color_space == 0u) {
        float yy = fmaxf(y_value - 16.0f, 0.0f) * 1.16438356f;
        r = yy + 1.59602678f * v_value;
        g = yy - 0.39176229f * u_value - 0.81296765f * v_value;
        blue = yy + 2.01723214f * u_value;
    } else if (color_space == 1u) {
        float yy = fmaxf(y_value - 16.0f, 0.0f) * 1.16438356f;
        r = yy + 1.79274107f * v_value;
        g = yy - 0.21324861f * u_value - 0.53290933f * v_value;
        blue = yy + 2.11240179f * u_value;
    } else if (color_space == 2u) {
        r = y_value + 1.402f * v_value;
        g = y_value - 0.344136f * u_value - 0.714136f * v_value;
        blue = y_value + 1.772f * u_value;
    } else {
        r = y_value + 1.5748f * v_value;
        g = y_value - 0.187324f * u_value - 0.468124f * v_value;
        blue = y_value + 1.8556f * u_value;
    }

    float values[3] = {
        swm_clamp(r, 0.0f, 255.0f) * 0.00392156862745098f,
        swm_clamp(g, 0.0f, 255.0f) * 0.00392156862745098f,
        swm_clamp(blue, 0.0f, 255.0f) * 0.00392156862745098f,
    };
    float means[3] = { mean0, mean1, mean2 };
    float stds[3] = { std0, std1, std2 };

    for (unsigned int c = 0u; c < 3u; c++) {
        float normalized = (values[c] - means[c]) / stds[c];
        unsigned int out_idx;
        if (history_len == 0u) {
            out_idx = (((b * 3u + c) * out_h + y) * out_w + x);
        } else {
            out_idx = ((((b * history_len + history_slot) * 3u + c) * out_h + y) * out_w + x);
        }
        output[out_idx] = normalized;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocesses_packed_rgb_to_nchw_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = PackedImageShape::new(1, 2, 2, PackedImageFormat::Rgb);
        let input = packed_u8_tensor_from_host(
            &[
                255, 0, 0, //
                0, 255, 0, //
                0, 0, 255, //
                255, 255, 255,
            ],
            shape,
            &device,
        )?;
        let config = ImagePreprocess {
            output_height: 2,
            output_width: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor = ImagePreprocessor::new(&device, shape, config)?;
        let output = preprocessor.preprocess_packed_u8(&input)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = [
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 1.0,
        ];

        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1e-6,
                "output[{idx}] expected {expected}, got {actual}, diff {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn preprocesses_packed_rgb_into_history_slot_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = PackedImageShape::new(1, 1, 1, PackedImageFormat::Bgr);
        let input = packed_u8_tensor_from_host(&[0, 128, 255], shape, &device)?;
        let config = ImagePreprocess {
            output_height: 1,
            output_width: 1,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor = ImageHistoryPreprocessor::new(&device, shape, 3, config)?;
        let output = preprocessor.preprocess_packed_u8_into_slot(&input, 1)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 128.0 / 255.0, 0.0, 0.0, 0.0, 0.0];

        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1e-6,
                "output[{idx}] expected {expected}, got {actual}, diff {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn preprocesses_nv12_to_nchw_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = Nv12ImageShape::new(1, 2, 2);
        let (y_plane, uv_plane) =
            nv12_tensors_from_host(&[0, 0, 0, 0], &[128, 128], shape, &device)?;
        let config = ImagePreprocess {
            output_height: 2,
            output_width: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor =
            Nv12Preprocessor::new(&device, shape, Nv12ColorSpace::Bt601Full, config)?;
        let output = preprocessor.preprocess_nv12(&y_plane, &uv_plane)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), 12);
        for (idx, actual) in actual.iter().enumerate() {
            assert!(
                actual.abs() <= 1e-6,
                "output[{idx}] expected 0.0, got {actual}"
            );
        }
        Ok(())
    }

    #[test]
    fn preprocesses_nv12_into_history_slot_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = Nv12ImageShape::new(1, 2, 2);
        let (y_plane, uv_plane) =
            nv12_tensors_from_host(&[255, 255, 255, 255], &[128, 128], shape, &device)?;
        let config = ImagePreprocess {
            output_height: 2,
            output_width: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor =
            Nv12HistoryPreprocessor::new(&device, shape, Nv12ColorSpace::Bt709Full, 2, config)?;
        let output = preprocessor.preprocess_nv12_into_slot(&y_plane, &uv_plane, 1)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual.len(), 24);

        for (idx, actual) in actual.iter().take(12).enumerate() {
            assert!(
                actual.abs() <= 1e-6,
                "output[{idx}] expected 0.0, got {actual}"
            );
        }
        for (offset, actual) in actual.iter().skip(12).enumerate() {
            let idx = offset + 12;
            let diff = (actual - 1.0).abs();
            assert!(
                diff <= 1e-6,
                "output[{idx}] expected 1.0, got {actual}, diff {diff}"
            );
        }
        Ok(())
    }
}
