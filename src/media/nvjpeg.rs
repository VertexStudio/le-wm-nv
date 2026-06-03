//! nvJPEG decode into Candle CUDA tensors.

use std::{fmt, os::raw::c_int, ptr::null_mut};

use candle::{
    CudaStorage, DType, Device, InplaceOp1, Layout, Result, Tensor, backend::BackendStorage,
    cuda::cudarc::driver::DevicePtrMut,
};
use nvjpeg_sys::{
    cudaStream_t, nvjpegChromaSubsampling_t, nvjpegCreateSimple, nvjpegDecode, nvjpegDestroy,
    nvjpegGetImageInfo, nvjpegHandle_t, nvjpegImage_t, nvjpegJpegState_t, nvjpegJpegStateCreate,
    nvjpegJpegStateDestroy, nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI, nvjpegStatus_t,
    nvjpegStatus_t_NVJPEG_STATUS_SUCCESS,
};

use super::{
    ImageHistoryPreprocessor, ImagePreprocess, ImagePreprocessor, PackedImageFormat,
    PackedImageShape, contiguous_slice_mut, require_contiguous,
};

pub struct NvJpegDecoder {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    device: Device,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvJpegImageInfo {
    pub width: usize,
    pub height: usize,
    pub components: usize,
    pub subsampling: nvjpegChromaSubsampling_t,
}

#[derive(Debug)]
pub struct NvJpegDecodedImage {
    pub tensor: Tensor,
    pub shape: PackedImageShape,
}

impl fmt::Debug for NvJpegDecoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NvJpegDecoder")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

impl NvJpegDecoder {
    pub fn new(device: &Device) -> Result<Self> {
        if !device.is_cuda() {
            candle::bail!("NvJpegDecoder requires a CUDA Candle device");
        }

        let mut handle: nvjpegHandle_t = null_mut();
        check_nvjpeg(
            unsafe { nvjpegCreateSimple(&mut handle) },
            "nvjpegCreateSimple",
        )?;

        let mut state: nvjpegJpegState_t = null_mut();
        let state_result = unsafe { nvjpegJpegStateCreate(handle, &mut state) };
        if let Err(err) = check_nvjpeg(state_result, "nvjpegJpegStateCreate") {
            unsafe {
                nvjpegDestroy(handle);
            }
            return Err(err);
        }

        Ok(Self {
            handle,
            state,
            device: device.clone(),
        })
    }

    pub fn image_info(&self, encoded: &[u8]) -> Result<NvJpegImageInfo> {
        require_encoded(encoded)?;

        let mut components: c_int = 0;
        let mut subsampling: nvjpegChromaSubsampling_t = 0;
        let mut widths = [0 as c_int; 4];
        let mut heights = [0 as c_int; 4];

        check_nvjpeg(
            unsafe {
                nvjpegGetImageInfo(
                    self.handle,
                    encoded.as_ptr(),
                    encoded.len(),
                    &mut components,
                    &mut subsampling,
                    widths.as_mut_ptr(),
                    heights.as_mut_ptr(),
                )
            },
            "nvjpegGetImageInfo",
        )?;

        if widths[0] <= 0 || heights[0] <= 0 {
            candle::bail!(
                "nvJPEG returned invalid image dimensions {}x{}",
                widths[0],
                heights[0]
            );
        }

        Ok(NvJpegImageInfo {
            width: widths[0] as usize,
            height: heights[0] as usize,
            components: components as usize,
            subsampling,
        })
    }

    pub fn alloc_rgb_interleaved(&self, info: NvJpegImageInfo) -> Result<Tensor> {
        validate_info(info)?;
        Tensor::zeros((1, info.height, info.width, 3), DType::U8, &self.device)
    }

    pub fn decode_rgb_interleaved(&mut self, encoded: &[u8]) -> Result<NvJpegDecodedImage> {
        let info = self.image_info(encoded)?;
        let tensor = self.alloc_rgb_interleaved(info)?;
        self.decode_rgb_interleaved_info_into(encoded, &tensor, info)?;
        Ok(NvJpegDecodedImage {
            tensor,
            shape: info.packed_rgb_shape(),
        })
    }

    pub fn decode_rgb_interleaved_into(
        &mut self,
        encoded: &[u8],
        output: &Tensor,
    ) -> Result<NvJpegImageInfo> {
        let info = self.image_info(encoded)?;
        self.decode_rgb_interleaved_info_into(encoded, output, info)?;
        Ok(info)
    }

    fn decode_rgb_interleaved_info_into(
        &mut self,
        encoded: &[u8],
        output: &Tensor,
        info: NvJpegImageInfo,
    ) -> Result<()> {
        validate_rgb_output(output, info)?;
        let op = NvJpegDecodeRgbInterleaved {
            handle: self.handle,
            state: self.state,
            encoded,
            info,
        };
        output.inplace_op1(&op)
    }

    pub fn decode_preprocessed_nchw(
        &mut self,
        encoded: &[u8],
        config: ImagePreprocess,
    ) -> Result<Tensor> {
        let decoded = self.decode_rgb_interleaved(encoded)?;
        let mut preprocessor = ImagePreprocessor::new(&self.device, decoded.shape, config)?;
        preprocessor
            .preprocess_packed_u8(&decoded.tensor)?
            .contiguous()
    }

    pub fn decode_preprocessed_nchw_into<'a>(
        &mut self,
        encoded: &[u8],
        rgb_output: &Tensor,
        preprocessor: &'a mut ImagePreprocessor,
    ) -> Result<&'a Tensor> {
        let info = self.image_info(encoded)?;
        require_preprocessor_shape(info, preprocessor.input_shape())?;
        self.decode_rgb_interleaved_info_into(encoded, rgb_output, info)?;
        preprocessor.preprocess_packed_u8(rgb_output)
    }

    pub fn decode_preprocessed_history_slot_into<'a>(
        &mut self,
        encoded: &[u8],
        rgb_output: &Tensor,
        history_slot: usize,
        preprocessor: &'a mut ImageHistoryPreprocessor,
    ) -> Result<&'a Tensor> {
        let info = self.image_info(encoded)?;
        require_preprocessor_shape(info, preprocessor.input_shape())?;
        self.decode_rgb_interleaved_info_into(encoded, rgb_output, info)?;
        preprocessor.preprocess_packed_u8_into_slot(rgb_output, history_slot)
    }
}

impl Drop for NvJpegDecoder {
    fn drop(&mut self) {
        unsafe {
            if !self.state.is_null() {
                nvjpegJpegStateDestroy(self.state);
            }
            if !self.handle.is_null() {
                nvjpegDestroy(self.handle);
            }
        }
    }
}

impl NvJpegImageInfo {
    pub fn packed_rgb_shape(self) -> PackedImageShape {
        PackedImageShape::new(1, self.height, self.width, PackedImageFormat::Rgb)
    }
}

struct NvJpegDecodeRgbInterleaved<'a> {
    handle: nvjpegHandle_t,
    state: nvjpegJpegState_t,
    encoded: &'a [u8],
    info: NvJpegImageInfo,
}

impl InplaceOp1 for NvJpegDecodeRgbInterleaved<'_> {
    fn name(&self) -> &'static str {
        "nvjpeg-decode-rgb-interleaved"
    }

    fn cpu_fwd(&self, _storage: &mut candle::CpuStorage, _layout: &Layout) -> Result<()> {
        candle::bail!("nvjpeg-decode-rgb-interleaved is a CUDA media operation")
    }

    fn cuda_fwd(&self, storage: &mut CudaStorage, layout: &Layout) -> Result<()> {
        if storage.dtype() != DType::U8 {
            candle::bail!("nvJPEG output tensor must use U8 dtype");
        }
        let expected = [1, self.info.height, self.info.width, 3];
        if layout.dims() != expected {
            candle::bail!(
                "nvJPEG output shape mismatch: expected {:?}, got {:?}",
                expected,
                layout.dims()
            );
        }
        require_contiguous(layout, "nvJPEG output")?;

        let stream = storage.device.cuda_stream();
        let mut output = storage.as_cuda_slice_mut::<u8>()?;
        let mut output = contiguous_slice_mut(&mut output, layout, "nvJPEG output")?;
        let (ptr, _record_write) = output.device_ptr_mut(&stream);

        let mut image = nvjpegImage_t::new();
        image.channel[0] = ptr as usize as *mut u8;
        image.pitch[0] = self.info.width * 3;

        check_nvjpeg(
            unsafe {
                nvjpegDecode(
                    self.handle,
                    self.state,
                    self.encoded.as_ptr(),
                    self.encoded.len(),
                    nvjpegOutputFormat_t_NVJPEG_OUTPUT_RGBI,
                    &mut image,
                    stream.cu_stream() as cudaStream_t,
                )
            },
            "nvjpegDecode",
        )
    }
}

fn require_encoded(encoded: &[u8]) -> Result<()> {
    if encoded.is_empty() {
        candle::bail!("encoded JPEG buffer is empty");
    }
    Ok(())
}

fn validate_info(info: NvJpegImageInfo) -> Result<()> {
    if info.width == 0 || info.height == 0 {
        candle::bail!(
            "nvJPEG image dimensions must be greater than zero, got {}x{}",
            info.width,
            info.height
        );
    }
    Ok(())
}

fn validate_rgb_output(output: &Tensor, info: NvJpegImageInfo) -> Result<()> {
    if !output.device().is_cuda() {
        candle::bail!("nvJPEG output tensor must live on a CUDA Candle device");
    }
    if output.dtype() != DType::U8 {
        candle::bail!("nvJPEG output tensor must use U8 dtype");
    }
    let expected = [1, info.height, info.width, 3];
    if output.dims() != expected {
        candle::bail!(
            "nvJPEG output shape mismatch: expected {:?}, got {:?}",
            expected,
            output.dims()
        );
    }
    Ok(())
}

fn require_preprocessor_shape(info: NvJpegImageInfo, shape: PackedImageShape) -> Result<()> {
    let expected = info.packed_rgb_shape();
    if shape != expected {
        candle::bail!(
            "preprocessor input shape mismatch: expected {:?}, got {:?}",
            expected,
            shape
        );
    }
    Ok(())
}

fn check_nvjpeg(status: nvjpegStatus_t, context: &str) -> Result<()> {
    if status == nvjpegStatus_t_NVJPEG_STATUS_SUCCESS {
        return Ok(());
    }
    candle::bail!(
        "{context} failed with nvJPEG status {status} ({})",
        status_name(status)
    )
}

fn status_name(status: nvjpegStatus_t) -> &'static str {
    match status {
        1 => "not initialized",
        2 => "invalid parameter",
        3 => "bad jpeg",
        4 => "jpeg not supported",
        5 => "allocator failure",
        6 => "execution failed",
        7 => "arch mismatch",
        8 => "internal error",
        9 => "implementation not supported",
        10 => "incomplete bitstream",
        _ => "unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_JPEG_1X1_RGB_HEX: &str = concat!(
        "ffd8ffe000104a46494600010100000100010000ffdb004300080606070605080707070909080a0c140d0c0b0b0c1912",
        "130f141d1a1f1e1d1a1c1c20242e2720222c231c1c2837292c30313434341f27393d38323c2e333432ffdb0043010909",
        "090c0b0c180d0d1832211c21323232323232323232323232323232323232323232323232323232323232323232323232",
        "3232323232323232323232323232ffc00011080001000103012200021101031101ffc4001f0000010501010101010100",
        "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221",
        "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a",
        "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899",
        "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1",
        "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100",
        "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015",
        "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869",
        "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4",
        "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00e2",
        "e8a28af993f713ffd9",
    );

    #[test]
    fn decodes_jpeg_to_cuda_rgb_tensor() -> Result<()> {
        let jpeg = decode_hex(TEST_JPEG_1X1_RGB_HEX);
        let device = Device::new_cuda(0)?;
        let mut decoder = NvJpegDecoder::new(&device)?;
        let info = decoder.image_info(&jpeg)?;
        assert_eq!((info.width, info.height), (1, 1));

        let decoded = decoder.decode_rgb_interleaved(&jpeg)?;
        assert_eq!(
            decoded.shape,
            PackedImageShape::new(1, 1, 1, PackedImageFormat::Rgb)
        );
        assert_eq!(decoded.tensor.dims(), &[1, 1, 1, 3]);

        let config = ImagePreprocess {
            output_height: 1,
            output_width: 1,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor = ImagePreprocessor::new(&device, decoded.shape, config)?;
        let output = preprocessor.preprocess_packed_u8(&decoded.tensor)?;
        assert_eq!(output.dims(), &[1, 3, 1, 1]);
        Ok(())
    }

    fn decode_hex(hex: &str) -> Vec<u8> {
        assert_eq!(hex.len() % 2, 0);
        (0..hex.len())
            .step_by(2)
            .map(|idx| u8::from_str_radix(&hex[idx..idx + 2], 16).unwrap())
            .collect()
    }
}
