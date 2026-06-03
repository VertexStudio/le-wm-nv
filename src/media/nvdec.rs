//! NVIDIA Video Decoder capability queries and CUDA NV12 decode sessions.
//!
//! This module binds the NVDECODE surface used by the runtime: device-scoped
//! capability checks, decoder/parser lifecycle, packet parsing, frame mapping,
//! and CUDA-side copies into Candle-compatible NV12 tensors.

use std::{
    ffi::{c_int, c_longlong, c_short, c_uchar, c_uint, c_ulong, c_ulonglong, c_ushort, c_void},
    fmt, ptr,
};

use candle::{
    Device, Result,
    cuda::{
        CudaDevice, WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
};

use super::{Nv12ImageShape, cuda_u8_tensor_device_ptr, validate_nv12_tensors};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
}

impl NvDecCodec {
    fn raw(self) -> u32 {
        match self {
            Self::H264 => 4,
            Self::Hevc => 8,
            Self::Vp9 => 10,
            Self::Av1 => 11,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

impl TryFrom<u32> for NvDecCodec {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::H264),
            1 => Ok(Self::Hevc),
            2 => Ok(Self::Av1),
            3 => Ok(Self::Vp9),
            other => candle::bail!(
                "unknown NVDECODE codec {other}; expected 0=H264, 1=HEVC, 2=AV1, or 3=VP9"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvDecCaps {
    pub codec: NvDecCodec,
    pub chroma_format: NvDecChromaFormat,
    pub bit_depth_minus_8: u32,
    pub supported: bool,
    pub nvdec_count: usize,
    pub output_format_mask: u32,
    pub min_width: usize,
    pub min_height: usize,
    pub max_width: usize,
    pub max_height: usize,
    pub max_macroblock_count: usize,
    pub histogram_supported: bool,
    pub histogram_counter_bit_depth: usize,
    pub max_histogram_bins: usize,
}

impl NvDecCaps {
    pub fn supports(self, format: NvDecSurfaceFormat) -> bool {
        self.output_format_mask & format.mask_bit() != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecChromaFormat {
    Monochrome,
    Yuv420,
    Yuv422,
    Yuv444,
}

impl NvDecChromaFormat {
    fn raw(self) -> u32 {
        match self {
            Self::Monochrome => 0,
            Self::Yuv420 => 1,
            Self::Yuv422 => 2,
            Self::Yuv444 => 3,
        }
    }
}

impl TryFrom<u32> for NvDecChromaFormat {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Monochrome),
            1 => Ok(Self::Yuv420),
            2 => Ok(Self::Yuv422),
            3 => Ok(Self::Yuv444),
            other => candle::bail!(
                "unknown NVDECODE chroma format {other}; expected 0=monochrome, 1=YUV420, 2=YUV422, or 3=YUV444"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecSurfaceFormat {
    Nv12,
    P016,
    Yuv444,
    Yuv44416Bit,
}

impl NvDecSurfaceFormat {
    fn raw(self) -> u32 {
        match self {
            Self::Nv12 => 0,
            Self::P016 => 1,
            Self::Yuv444 => 2,
            Self::Yuv44416Bit => 3,
        }
    }

    fn mask_bit(self) -> u32 {
        1 << self.raw()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvDecDecoderConfig {
    pub codec: NvDecCodec,
    pub width: usize,
    pub height: usize,
    pub decode_surfaces: usize,
    pub output_surfaces: usize,
}

impl NvDecDecoderConfig {
    pub fn new(codec: NvDecCodec, width: usize, height: usize) -> Self {
        Self {
            codec,
            width,
            height,
            decode_surfaces: 20,
            output_surfaces: 2,
        }
    }

    fn validate(self) -> Result<()> {
        if self.width == 0 || self.height == 0 {
            candle::bail!("NVDECODE decoder dimensions must be greater than zero");
        }
        if self.width % 2 != 0 || self.height % 2 != 0 {
            candle::bail!(
                "NVDECODE NV12 decoder dimensions must be even, got {}x{}",
                self.width,
                self.height
            );
        }
        if self.width > c_short::MAX as usize || self.height > c_short::MAX as usize {
            candle::bail!(
                "NVDECODE decoder dimensions exceed SDK rect range: {}x{}",
                self.width,
                self.height
            );
        }
        if self.decode_surfaces == 0 || self.output_surfaces == 0 {
            candle::bail!("NVDECODE decoder surface counts must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct NvDecDecoder {
    decoder: CuVideoDecoder,
    ctx_lock: CuVideoCtxLock,
    config: NvDecDecoderConfig,
}

impl NvDecDecoder {
    pub fn new_nv12(device: &Device, config: NvDecDecoderConfig) -> Result<Self> {
        config.validate()?;

        let caps = query_caps_420(device, config.codec, 0)?;
        if !caps.supported {
            candle::bail!(
                "NVDECODE {} 8-bit 4:2:0 is not supported on the selected CUDA device",
                config.codec.name()
            );
        }
        if !caps.supports(NvDecSurfaceFormat::Nv12) {
            candle::bail!(
                "NVDECODE {} 8-bit 4:2:0 does not report NV12 output support",
                config.codec.name()
            );
        }
        if config.width < caps.min_width
            || config.height < caps.min_height
            || config.width > caps.max_width
            || config.height > caps.max_height
        {
            candle::bail!(
                "NVDECODE {} dimensions {}x{} outside supported range {}x{}..{}x{}",
                config.codec.name(),
                config.width,
                config.height,
                caps.min_width,
                caps.min_height,
                caps.max_width,
                caps.max_height
            );
        }

        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let context = stream.context();
        context.bind_to_thread().w()?;

        let mut ctx_lock = ptr::null_mut();
        check_cuvid(
            unsafe {
                cuvidCtxLockCreate(
                    &mut ctx_lock as *mut CuVideoCtxLock,
                    context.cu_ctx() as CuContext,
                )
            },
            "cuvidCtxLockCreate",
        )?;

        let rect = CuvidRect {
            left: 0,
            top: 0,
            right: config.width as c_short,
            bottom: config.height as c_short,
        };
        let mut create_info = CuvidDecodeCreateInfo {
            width: config.width as c_ulong,
            height: config.height as c_ulong,
            decode_surfaces: config.decode_surfaces as c_ulong,
            codec_type: config.codec.raw(),
            chroma_format: NvDecChromaFormat::Yuv420.raw(),
            creation_flags: CUDA_VIDEO_CREATE_PREFER_CUVID as c_ulong,
            bit_depth_minus_8: 0,
            intra_decode_only: 0,
            max_width: config.width as c_ulong,
            max_height: config.height as c_ulong,
            reserved1: 0,
            display_area: rect,
            output_format: NvDecSurfaceFormat::Nv12.raw(),
            deinterlace_mode: CUDA_VIDEO_DEINTERLACE_WEAVE,
            target_width: config.width as c_ulong,
            target_height: config.height as c_ulong,
            output_surfaces: config.output_surfaces as c_ulong,
            ctx_lock,
            target_rect: rect,
            enable_histogram: 0,
            reserved2: [0; 4],
        };

        let mut decoder = ptr::null_mut();
        if let Err(err) = check_cuvid(
            unsafe {
                cuvidCreateDecoder(
                    &mut decoder as *mut CuVideoDecoder,
                    &mut create_info as *mut CuvidDecodeCreateInfo,
                )
            },
            "cuvidCreateDecoder",
        ) {
            unsafe {
                cuvidCtxLockDestroy(ctx_lock);
            }
            return Err(err);
        }

        Ok(Self {
            decoder,
            ctx_lock,
            config,
        })
    }

    pub fn config(&self) -> NvDecDecoderConfig {
        self.config
    }

    fn raw_decoder(&self) -> CuVideoDecoder {
        self.decoder
    }
}

impl Drop for NvDecDecoder {
    fn drop(&mut self) {
        if !self.decoder.is_null() {
            unsafe {
                cuvidDestroyDecoder(self.decoder);
            }
            self.decoder = ptr::null_mut();
        }
        if !self.ctx_lock.is_null() {
            unsafe {
                cuvidCtxLockDestroy(self.ctx_lock);
            }
            self.ctx_lock = ptr::null_mut();
        }
    }
}

pub fn query_caps_420(
    device: &Device,
    codec: NvDecCodec,
    bit_depth_minus_8: u32,
) -> Result<NvDecCaps> {
    query_caps(device, codec, NvDecChromaFormat::Yuv420, bit_depth_minus_8)
}

pub fn query_caps(
    device: &Device,
    codec: NvDecCodec,
    chroma_format: NvDecChromaFormat,
    bit_depth_minus_8: u32,
) -> Result<NvDecCaps> {
    if !device.is_cuda() {
        candle::bail!("NVDECODE capability query requires a CUDA Candle device");
    }
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    stream.context().bind_to_thread().w()?;

    let mut raw = CuvidDecodeCaps {
        e_codec_type: codec.raw(),
        e_chroma_format: chroma_format.raw(),
        n_bit_depth_minus_8: bit_depth_minus_8,
        ..Default::default()
    };
    check_cuvid(
        unsafe { cuvidGetDecoderCaps(&mut raw as *mut CuvidDecodeCaps) },
        "cuvidGetDecoderCaps",
    )?;

    Ok(NvDecCaps {
        codec,
        chroma_format,
        bit_depth_minus_8,
        supported: raw.b_is_supported != 0,
        nvdec_count: raw.n_num_nvdecs as usize,
        output_format_mask: raw.n_output_format_mask as u32,
        min_width: raw.n_min_width as usize,
        min_height: raw.n_min_height as usize,
        max_width: raw.n_max_width as usize,
        max_height: raw.n_max_height as usize,
        max_macroblock_count: raw.n_max_mb_count as usize,
        histogram_supported: raw.b_is_histogram_supported != 0,
        histogram_counter_bit_depth: raw.n_counter_bit_depth as usize,
        max_histogram_bins: raw.n_max_histogram_bins as usize,
    })
}

#[derive(Debug)]
pub struct NvDecSession {
    decoder: NvDecDecoder,
    parser: CuVideoParser,
    state: Box<NvDecParserState>,
}

impl NvDecSession {
    pub fn new_nv12(device: &Device, config: NvDecDecoderConfig) -> Result<Self> {
        let decoder = NvDecDecoder::new_nv12(device, config)?;
        let cuda = device.as_cuda_device()?.clone();
        let mut state = Box::new(NvDecParserState::new(decoder.raw_decoder(), cuda, config));

        let mut params = CuvidParserParams {
            codec_type: config.codec.raw(),
            max_num_decode_surfaces: config.decode_surfaces as c_uint,
            clock_rate: 1000,
            error_threshold: 100,
            max_display_delay: 0,
            annexb_and_reserved: 1,
            reserved1: [0; 4],
            user_data: state.as_mut() as *mut NvDecParserState as *mut c_void,
            sequence_callback: Some(nvdec_sequence_callback),
            decode_picture_callback: Some(nvdec_decode_picture_callback),
            display_picture_callback: Some(nvdec_display_picture_callback),
            operating_point_callback: None,
            sei_message_callback: None,
            reserved2: [ptr::null_mut(); 5],
            ext_video_info: ptr::null_mut(),
        };
        let mut parser = ptr::null_mut();
        check_cuvid(
            unsafe {
                cuvidCreateVideoParser(
                    &mut parser as *mut CuVideoParser,
                    &mut params as *mut CuvidParserParams,
                )
            },
            "cuvidCreateVideoParser",
        )?;

        Ok(Self {
            decoder,
            parser,
            state,
        })
    }

    pub fn config(&self) -> NvDecDecoderConfig {
        self.decoder.config()
    }

    pub fn decode_annexb_to_nv12(
        &mut self,
        encoded: &[u8],
        y_plane: &candle::Tensor,
        uv_plane: &candle::Tensor,
    ) -> Result<usize> {
        if encoded.is_empty() {
            candle::bail!("NVDECODE encoded packet is empty");
        }
        let shape = Nv12ImageShape::new(1, self.config().height, self.config().width);
        validate_nv12_tensors(y_plane, uv_plane, shape)?;
        let y_ptr = cuda_u8_tensor_device_ptr(y_plane)? as usize as u64;
        let uv_ptr = cuda_u8_tensor_device_ptr(uv_plane)? as usize as u64;
        self.state.begin_decode(Nv12DecodeTarget {
            y_ptr,
            uv_ptr,
            width: shape.width,
            height: shape.height,
        });

        let mut packet = CuvidSourceDataPacket {
            flags: CUVID_PKT_ENDOFPICTURE as c_ulong,
            payload_size: encoded.len() as c_ulong,
            payload: encoded.as_ptr(),
            timestamp: 0,
        };
        let parse_result = check_cuvid(
            unsafe { cuvidParseVideoData(self.parser, &mut packet as *mut CuvidSourceDataPacket) },
            "cuvidParseVideoData",
        );
        let frames = self.state.finish_decode()?;
        parse_result?;
        Ok(frames)
    }
}

impl Drop for NvDecSession {
    fn drop(&mut self) {
        if !self.parser.is_null() {
            unsafe {
                cuvidDestroyVideoParser(self.parser);
            }
            self.parser = ptr::null_mut();
        }
    }
}

#[derive(Debug)]
struct NvDecParserState {
    decoder: CuVideoDecoder,
    device: CudaDevice,
    config: NvDecDecoderConfig,
    target: Option<Nv12DecodeTarget>,
    frames_decoded: usize,
    error: Option<String>,
}

impl NvDecParserState {
    fn new(decoder: CuVideoDecoder, device: CudaDevice, config: NvDecDecoderConfig) -> Self {
        Self {
            decoder,
            device,
            config,
            target: None,
            frames_decoded: 0,
            error: None,
        }
    }

    fn begin_decode(&mut self, target: Nv12DecodeTarget) {
        self.target = Some(target);
        self.frames_decoded = 0;
        self.error = None;
    }

    fn finish_decode(&mut self) -> Result<usize> {
        self.target = None;
        if let Some(error) = self.error.take() {
            candle::bail!("{error}");
        }
        Ok(self.frames_decoded)
    }

    fn set_error(&mut self, error: impl fmt::Display) {
        self.error = Some(error.to_string());
    }

    fn copy_mapped_nv12(&mut self, src_ptr: u64, src_pitch: usize) -> Result<()> {
        let Some(target) = self.target else {
            candle::bail!("NVDECODE display callback has no CUDA NV12 output target");
        };
        if target.width != self.config.width || target.height != self.config.height {
            candle::bail!(
                "NVDECODE output target {}x{} does not match decoder {}x{}",
                target.width,
                target.height,
                self.config.width,
                self.config.height
            );
        }
        if src_pitch < target.width {
            candle::bail!(
                "NVDECODE mapped frame pitch {src_pitch} is smaller than width {}",
                target.width
            );
        }

        let ptx = nvrtc::safe::compile_ptx_with_opts(
            NVDEC_COPY_NV12_CUDA,
            nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .w()?;
        let func = self.device.get_or_load_custom_func(
            "swm_copy_nvdec_nv12",
            "swm_nvdec_copy",
            &ptx.to_src(),
        )?;

        let byte_count = target.width * target.height * 3 / 2;
        let cfg = LaunchConfig::for_num_elems(byte_count as u32);
        let byte_count_u32 = byte_count as u32;
        let src_pitch_u32 = src_pitch as u32;
        let width_u32 = target.width as u32;
        let height_u32 = target.height as u32;
        let mut builder = func.builder();
        builder.arg(&src_ptr);
        builder.arg(&src_pitch_u32);
        builder.arg(&target.y_ptr);
        builder.arg(&target.uv_ptr);
        builder.arg(&byte_count_u32);
        builder.arg(&width_u32);
        builder.arg(&height_u32);
        unsafe { builder.launch(cfg) }.w()?;
        self.device.cuda_stream().synchronize().w()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy)]
struct Nv12DecodeTarget {
    y_ptr: u64,
    uv_ptr: u64,
    width: usize,
    height: usize,
}

unsafe extern "C" fn nvdec_sequence_callback(
    user_data: *mut c_void,
    format: *mut CuVideoFormat,
) -> c_int {
    let Some(state) = (unsafe { (user_data as *mut NvDecParserState).as_mut() }) else {
        return 0;
    };
    if format.is_null() {
        state.set_error("NVDECODE sequence callback received null format");
        return 0;
    }
    let format = unsafe { &*format };
    if format.coded_width as usize != state.config.width
        || format.coded_height as usize != state.config.height
    {
        let coded_width = format.coded_width;
        let coded_height = format.coded_height;
        let config_width = state.config.width;
        let config_height = state.config.height;
        state.set_error(format_args!(
            "NVDECODE bitstream dimensions {}x{} do not match decoder {}x{}",
            coded_width, coded_height, config_width, config_height
        ));
        return 0;
    }
    state
        .config
        .decode_surfaces
        .max(format.min_num_decode_surfaces as usize) as c_int
}

unsafe extern "C" fn nvdec_decode_picture_callback(
    user_data: *mut c_void,
    pic_params: *mut c_void,
) -> c_int {
    let Some(state) = (unsafe { (user_data as *mut NvDecParserState).as_mut() }) else {
        return 0;
    };
    if pic_params.is_null() {
        state.set_error("NVDECODE decode callback received null picture params");
        return 0;
    }
    match check_cuvid(
        unsafe { cuvidDecodePicture(state.decoder, pic_params) },
        "cuvidDecodePicture",
    ) {
        Ok(()) => 1,
        Err(err) => {
            state.set_error(err);
            0
        }
    }
}

unsafe extern "C" fn nvdec_display_picture_callback(
    user_data: *mut c_void,
    display: *mut CuvidParserDispInfo,
) -> c_int {
    let Some(state) = (unsafe { (user_data as *mut NvDecParserState).as_mut() }) else {
        return 0;
    };
    if display.is_null() {
        state.set_error("NVDECODE display callback received null display info");
        return 0;
    }
    let display = unsafe { &*display };
    let stream = state.device.cuda_stream();
    let mut proc_params = CuvidProcParams {
        progressive_frame: display.progressive_frame,
        second_field: 0,
        top_field_first: display.top_field_first,
        unpaired_field: 0,
        reserved_flags: 0,
        reserved_zero: 0,
        raw_input_dptr: 0,
        raw_input_pitch: 0,
        raw_input_format: 0,
        raw_output_dptr: 0,
        raw_output_pitch: 0,
        reserved1: 0,
        output_stream: stream.cu_stream() as CuStream,
        reserved: [0; 46],
        histogram_dptr: ptr::null_mut(),
        reserved2: [ptr::null_mut(); 1],
    };
    let mut mapped_ptr: c_ulonglong = 0;
    let mut pitch: c_uint = 0;
    let map_result = check_cuvid(
        unsafe {
            cuvidMapVideoFrame64(
                state.decoder,
                display.picture_index,
                &mut mapped_ptr as *mut c_ulonglong,
                &mut pitch as *mut c_uint,
                &mut proc_params as *mut CuvidProcParams,
            )
        },
        "cuvidMapVideoFrame64",
    );
    if let Err(err) = map_result {
        state.set_error(err);
        return 0;
    }

    let copy_result = state.copy_mapped_nv12(mapped_ptr, pitch as usize);
    let unmap_result = check_cuvid(
        unsafe { cuvidUnmapVideoFrame64(state.decoder, mapped_ptr) },
        "cuvidUnmapVideoFrame64",
    );
    match copy_result.and(unmap_result) {
        Ok(()) => {
            state.frames_decoded += 1;
            1
        }
        Err(err) => {
            state.set_error(err);
            0
        }
    }
}

fn check_cuvid(status: c_int, context: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        candle::bail!("{context} failed with CUDA driver status {status}")
    }
}

const CUDA_SUCCESS: c_int = 0;
const CUDA_VIDEO_DEINTERLACE_WEAVE: c_uint = 0;
const CUDA_VIDEO_CREATE_PREFER_CUVID: c_uint = 4;
const CUVID_PKT_ENDOFPICTURE: c_uint = 8;

type CuContext = *mut c_void;
type CuVideoDecoder = *mut c_void;
type CuVideoCtxLock = *mut c_void;
type CuVideoParser = *mut c_void;
type CuStream = *mut c_void;
type CuVideoTimestamp = c_longlong;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidDecodeCaps {
    e_codec_type: c_uint,
    e_chroma_format: c_uint,
    n_bit_depth_minus_8: c_uint,
    reserved1: [c_uint; 3],
    b_is_supported: c_uchar,
    n_num_nvdecs: c_uchar,
    n_output_format_mask: c_ushort,
    n_max_width: c_uint,
    n_max_height: c_uint,
    n_max_mb_count: c_uint,
    n_min_width: c_ushort,
    n_min_height: c_ushort,
    b_is_histogram_supported: c_uchar,
    n_counter_bit_depth: c_uchar,
    n_max_histogram_bins: c_ushort,
    reserved3: [c_uint; 10],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidRect {
    left: c_short,
    top: c_short,
    right: c_short,
    bottom: c_short,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidDecodeCreateInfo {
    width: c_ulong,
    height: c_ulong,
    decode_surfaces: c_ulong,
    codec_type: c_uint,
    chroma_format: c_uint,
    creation_flags: c_ulong,
    bit_depth_minus_8: c_ulong,
    intra_decode_only: c_ulong,
    max_width: c_ulong,
    max_height: c_ulong,
    reserved1: c_ulong,
    display_area: CuvidRect,
    output_format: c_uint,
    deinterlace_mode: c_uint,
    target_width: c_ulong,
    target_height: c_ulong,
    output_surfaces: c_ulong,
    ctx_lock: CuVideoCtxLock,
    target_rect: CuvidRect,
    enable_histogram: c_ulong,
    reserved2: [c_ulong; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidSourceDataPacket {
    flags: c_ulong,
    payload_size: c_ulong,
    payload: *const c_uchar,
    timestamp: CuVideoTimestamp,
}

type NvDecSequenceCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, format: *mut CuVideoFormat) -> c_int>;
type NvDecDecodeCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, pic_params: *mut c_void) -> c_int>;
type NvDecDisplayCallback = Option<
    unsafe extern "C" fn(user_data: *mut c_void, display: *mut CuvidParserDispInfo) -> c_int,
>;
type NvDecOperatingPointCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, info: *mut c_void) -> c_int>;
type NvDecSeiMessageCallback =
    Option<unsafe extern "C" fn(user_data: *mut c_void, info: *mut c_void) -> c_int>;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidParserParams {
    codec_type: c_uint,
    max_num_decode_surfaces: c_uint,
    clock_rate: c_uint,
    error_threshold: c_uint,
    max_display_delay: c_uint,
    annexb_and_reserved: c_uint,
    reserved1: [c_uint; 4],
    user_data: *mut c_void,
    sequence_callback: NvDecSequenceCallback,
    decode_picture_callback: NvDecDecodeCallback,
    display_picture_callback: NvDecDisplayCallback,
    operating_point_callback: NvDecOperatingPointCallback,
    sei_message_callback: NvDecSeiMessageCallback,
    reserved2: [*mut c_void; 5],
    ext_video_info: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidParserDispInfo {
    picture_index: c_int,
    progressive_frame: c_int,
    top_field_first: c_int,
    repeat_first_field: c_int,
    timestamp: CuVideoTimestamp,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuVideoFormat {
    codec: c_uint,
    frame_rate: CuVideoFormatFrameRate,
    progressive_sequence: c_uchar,
    bit_depth_luma_minus8: c_uchar,
    bit_depth_chroma_minus8: c_uchar,
    min_num_decode_surfaces: c_uchar,
    coded_width: c_uint,
    coded_height: c_uint,
    display_area: CuVideoFormatDisplayArea,
    chroma_format: c_uint,
    bitrate: c_uint,
    display_aspect_ratio: CuVideoFormatAspectRatio,
    video_signal_description: CuVideoSignalDescription,
    seqhdr_data_length: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuVideoFormatFrameRate {
    numerator: c_uint,
    denominator: c_uint,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuVideoFormatDisplayArea {
    left: c_int,
    top: c_int,
    right: c_int,
    bottom: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuVideoFormatAspectRatio {
    x: c_int,
    y: c_int,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuVideoSignalDescription {
    flags: c_uchar,
    color_primaries: c_uchar,
    transfer_characteristics: c_uchar,
    matrix_coefficients: c_uchar,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidProcParams {
    progressive_frame: c_int,
    second_field: c_int,
    top_field_first: c_int,
    unpaired_field: c_int,
    reserved_flags: c_uint,
    reserved_zero: c_uint,
    raw_input_dptr: c_ulonglong,
    raw_input_pitch: c_uint,
    raw_input_format: c_uint,
    raw_output_dptr: c_ulonglong,
    raw_output_pitch: c_uint,
    reserved1: c_uint,
    output_stream: CuStream,
    reserved: [c_uint; 46],
    histogram_dptr: *mut c_ulonglong,
    reserved2: [*mut c_void; 1],
}

impl Default for CuvidDecodeCaps {
    fn default() -> Self {
        Self {
            e_codec_type: 0,
            e_chroma_format: 0,
            n_bit_depth_minus_8: 0,
            reserved1: [0; 3],
            b_is_supported: 0,
            n_num_nvdecs: 0,
            n_output_format_mask: 0,
            n_max_width: 0,
            n_max_height: 0,
            n_max_mb_count: 0,
            n_min_width: 0,
            n_min_height: 0,
            b_is_histogram_supported: 0,
            n_counter_bit_depth: 0,
            n_max_histogram_bins: 0,
            reserved3: [0; 10],
        }
    }
}

#[link(name = "nvcuvid")]
unsafe extern "C" {
    fn cuvidGetDecoderCaps(caps: *mut CuvidDecodeCaps) -> c_int;
    fn cuvidCreateVideoParser(parser: *mut CuVideoParser, params: *mut CuvidParserParams) -> c_int;
    fn cuvidParseVideoData(parser: CuVideoParser, packet: *mut CuvidSourceDataPacket) -> c_int;
    fn cuvidDestroyVideoParser(parser: CuVideoParser) -> c_int;
    fn cuvidCreateDecoder(
        decoder: *mut CuVideoDecoder,
        create_info: *mut CuvidDecodeCreateInfo,
    ) -> c_int;
    fn cuvidDestroyDecoder(decoder: CuVideoDecoder) -> c_int;
    fn cuvidDecodePicture(decoder: CuVideoDecoder, pic_params: *mut c_void) -> c_int;
    fn cuvidMapVideoFrame64(
        decoder: CuVideoDecoder,
        picture_index: c_int,
        dev_ptr: *mut c_ulonglong,
        pitch: *mut c_uint,
        proc_params: *mut CuvidProcParams,
    ) -> c_int;
    fn cuvidUnmapVideoFrame64(decoder: CuVideoDecoder, dev_ptr: c_ulonglong) -> c_int;
    fn cuvidCtxLockCreate(ctx_lock: *mut CuVideoCtxLock, context: CuContext) -> c_int;
    fn cuvidCtxLockDestroy(ctx_lock: CuVideoCtxLock) -> c_int;
}

const NVDEC_COPY_NV12_CUDA: &str = r#"
extern "C" __global__ void swm_copy_nvdec_nv12(
    unsigned long long src_base,
    unsigned int src_pitch,
    unsigned long long y_dst_base,
    unsigned long long uv_dst_base,
    unsigned int byte_count,
    unsigned int width,
    unsigned int height
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= byte_count) {
        return;
    }

    const unsigned char* src = (const unsigned char*)src_base;
    unsigned char* y_dst = (unsigned char*)y_dst_base;
    unsigned char* uv_dst = (unsigned char*)uv_dst_base;
    unsigned int y_count = width * height;
    if (idx < y_count) {
        unsigned int row = idx / width;
        unsigned int col = idx - row * width;
        y_dst[idx] = src[row * src_pitch + col];
    } else {
        unsigned int uv_idx = idx - y_count;
        unsigned int row = uv_idx / width;
        unsigned int col = uv_idx - row * width;
        uv_dst[uv_idx] = src[height * src_pitch + row * src_pitch + col];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_h264_420_8bit_caps_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let caps = query_caps_420(&device, NvDecCodec::H264, 0)?;

        println!("NVDECODE {} 8-bit 4:2:0 caps: {caps:?}", caps.codec.name());
        assert!(caps.supported);
        assert!(caps.nvdec_count > 0);
        assert!(caps.max_width >= caps.min_width);
        assert!(caps.max_height >= caps.min_height);
        assert!(caps.supports(NvDecSurfaceFormat::Nv12));
        Ok(())
    }

    #[test]
    fn unsupported_codec_ids_are_rejected() {
        let err = NvDecCodec::try_from(99).unwrap_err();
        assert!(err.to_string().contains("unknown NVDECODE codec"));
    }

    #[test]
    fn create_info_matches_sdk_layout() {
        assert_eq!(std::mem::size_of::<CuvidDecodeCreateInfo>(), 176);
        assert_eq!(std::mem::align_of::<CuvidDecodeCreateInfo>(), 8);
    }

    #[test]
    fn parser_structs_match_sdk_layout() {
        assert_eq!(std::mem::size_of::<CuvidSourceDataPacket>(), 32);
        assert_eq!(std::mem::align_of::<CuvidSourceDataPacket>(), 8);
        assert_eq!(std::mem::size_of::<CuVideoFormat>(), 64);
        assert_eq!(std::mem::align_of::<CuVideoFormat>(), 4);
        assert_eq!(std::mem::size_of::<CuvidParserParams>(), 136);
        assert_eq!(std::mem::align_of::<CuvidParserParams>(), 8);
        assert_eq!(std::mem::size_of::<CuvidProcParams>(), 264);
        assert_eq!(std::mem::align_of::<CuvidProcParams>(), 8);
    }

    #[test]
    fn creates_and_destroys_h264_nv12_decoder_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = NvDecDecoderConfig::new(NvDecCodec::H264, 64, 64);
        let decoder = NvDecDecoder::new_nv12(&device, config)?;

        assert_eq!(decoder.config(), config);
        Ok(())
    }

    #[test]
    fn creates_and_destroys_h264_nv12_session_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let config = NvDecDecoderConfig::new(NvDecCodec::H264, 64, 64);
        let session = NvDecSession::new_nv12(&device, config)?;

        assert_eq!(session.config(), config);
        Ok(())
    }

    #[test]
    fn decodes_annexb_packet_from_env_to_nv12_on_cuda() -> Result<()> {
        let Some(path) = std::env::var_os("SWM_NVDEC_TEST_PACKET") else {
            return Ok(());
        };
        let encoded = std::fs::read(path).map_err(candle::Error::wrap)?;
        let device = Device::new_cuda(0)?;
        let config = NvDecDecoderConfig::new(NvDecCodec::H264, 64, 64);
        let mut session = NvDecSession::new_nv12(&device, config)?;
        let shape = Nv12ImageShape::new(1, 64, 64);
        let (y_plane, uv_plane) = crate::media::nv12_tensors(shape, &device)?;

        let frames = session.decode_annexb_to_nv12(&encoded, &y_plane, &uv_plane)?;

        assert!(frames >= 1);
        assert_eq!(y_plane.dims(), &[1, 64, 64]);
        assert_eq!(uv_plane.dims(), &[1, 32, 32, 2]);
        Ok(())
    }

    #[test]
    fn rejects_odd_nv12_decoder_dimensions() {
        let config = NvDecDecoderConfig::new(NvDecCodec::H264, 63, 64);
        let err = config.validate().unwrap_err();
        assert!(
            err.to_string()
                .contains("NVDECODE NV12 decoder dimensions must be even")
        );
    }
}
