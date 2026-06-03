use std::path::Path;

use candle::{DType, Device, Result};
use candle_nn::{VarBuilder, VarMap};

pub fn var_builder_from_path(
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[path], dtype, device) }
    } else {
        VarBuilder::from_pth(path, dtype, device)
    }
}

pub fn empty_var_builder(dtype: DType, device: &Device) -> VarBuilder<'static> {
    let vars = VarMap::new();
    VarBuilder::from_varmap(&vars, dtype, device)
}
