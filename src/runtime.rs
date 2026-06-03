//! Shared runtime options for selecting Candle devices and dtypes.

use std::{fmt, str::FromStr};

use candle::{DType, Device, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSpec {
    Cuda(usize),
}

impl DeviceSpec {
    pub fn resolve(self) -> Result<Device> {
        match self {
            Self::Cuda(index) => cuda_device(index),
        }
    }
}

impl Default for DeviceSpec {
    fn default() -> Self {
        Self::Cuda(0)
    }
}

impl fmt::Display for DeviceSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(index) => write!(f, "cuda:{index}"),
        }
    }
}

impl FromStr for DeviceSpec {
    type Err = String;

    fn from_str(input: &str) -> std::result::Result<Self, Self::Err> {
        let input = input.trim().to_ascii_lowercase();
        match input.as_str() {
            "cuda" => return Ok(Self::Cuda(0)),
            _ => {}
        }

        if let Some(index) = parse_index(&input, "cuda:")? {
            return Ok(Self::Cuda(index));
        }

        Err(format!(
            "unsupported device '{input}', expected cuda or cuda:<index>"
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DTypeSpec {
    F32,
    Bf16,
    F16,
}

impl DTypeSpec {
    pub fn dtype(self) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::Bf16 => DType::BF16,
            Self::F16 => DType::F16,
        }
    }
}

impl Default for DTypeSpec {
    fn default() -> Self {
        Self::F32
    }
}

impl fmt::Display for DTypeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => f.write_str("f32"),
            Self::Bf16 => f.write_str("bf16"),
            Self::F16 => f.write_str("f16"),
        }
    }
}

impl FromStr for DTypeSpec {
    type Err = String;

    fn from_str(input: &str) -> std::result::Result<Self, Self::Err> {
        match input.trim().to_ascii_lowercase().as_str() {
            "f32" | "float32" => Ok(Self::F32),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "f16" | "float16" => Ok(Self::F16),
            other => Err(format!(
                "unsupported dtype '{other}', expected f32, bf16, or f16"
            )),
        }
    }
}

fn parse_index(input: &str, prefix: &str) -> std::result::Result<Option<usize>, String> {
    let Some(raw) = input.strip_prefix(prefix) else {
        return Ok(None);
    };
    if raw.is_empty() {
        return Err(format!("missing device index after '{prefix}'"));
    }
    raw.parse::<usize>()
        .map(Some)
        .map_err(|_| format!("invalid device index '{raw}' after '{prefix}'"))
}

fn cuda_device(index: usize) -> Result<Device> {
    Device::new_cuda(index)
}
