use std::str::FromStr;

use candle::DType;
use le_wm_nv::runtime::{DTypeSpec, DeviceSpec};

#[test]
fn parses_device_specs() {
    assert_eq!(DeviceSpec::from_str("cuda").unwrap(), DeviceSpec::Cuda(0));
    assert_eq!(DeviceSpec::from_str("cuda:2").unwrap(), DeviceSpec::Cuda(2));
}

#[test]
fn rejects_invalid_device_specs() {
    assert!(DeviceSpec::from_str("cpu").is_err());
    assert!(DeviceSpec::from_str("cuda:").is_err());
    assert!(DeviceSpec::from_str("cuda:abc").is_err());
    assert!(DeviceSpec::from_str("metal").is_err());
    assert!(DeviceSpec::from_str("gpu").is_err());
}

#[test]
fn parses_dtype_specs() {
    assert_eq!(DTypeSpec::from_str("f32").unwrap().dtype(), DType::F32);
    assert_eq!(DTypeSpec::from_str("float32").unwrap().dtype(), DType::F32);
    assert_eq!(DTypeSpec::from_str("bf16").unwrap().dtype(), DType::BF16);
    assert_eq!(
        DTypeSpec::from_str("bfloat16").unwrap().dtype(),
        DType::BF16
    );
    assert_eq!(DTypeSpec::from_str("f16").unwrap().dtype(), DType::F16);
    assert_eq!(DTypeSpec::from_str("float16").unwrap().dtype(), DType::F16);
}

#[test]
fn rejects_invalid_dtype_specs() {
    assert!(DTypeSpec::from_str("i8").is_err());
    assert!(DTypeSpec::from_str("").is_err());
}
