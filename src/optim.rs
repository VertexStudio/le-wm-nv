use std::{collections::HashMap, path::Path};

use candle::{Result, Tensor, Var};
use candle_nn::{ParamsAdamW, VarMap};

#[derive(Debug)]
struct VarAdamW {
    name: String,
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct StatefulAdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    params: ParamsAdamW,
}

impl StatefulAdamW {
    pub fn new_from_varmap(varmap: &VarMap, params: ParamsAdamW) -> Result<Self> {
        let mut named_vars = varmap
            .data()
            .lock()
            .unwrap()
            .iter()
            .filter(|(_, var)| var.dtype().is_float())
            .map(|(name, var)| (name.clone(), var.clone()))
            .collect::<Vec<_>>();
        named_vars.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        let vars = named_vars
            .into_iter()
            .map(|(name, var)| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    name,
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    /// Backpropagates, applies global L2 gradient clipping on-device, and
    /// performs one AdamW update. The returned norm is the pre-clipping norm.
    pub fn backward_step_clipped(&mut self, loss: &Tensor, max_norm: f64) -> Result<f32> {
        if !max_norm.is_finite() || max_norm <= 0.0 {
            candle::bail!("gradient max_norm must be finite and positive");
        }
        let mut grads = loss.backward()?;
        let mut norm_sq: Option<Tensor> = None;
        for var in &self.vars {
            if let Some(grad) = grads.get(var.var.as_tensor()) {
                let value = grad.sqr()?.sum_all()?;
                norm_sq = Some(match norm_sq {
                    Some(total) => (total + value)?,
                    None => value,
                });
            }
        }
        let norm = norm_sq
            .ok_or_else(|| candle::Error::Msg("loss produced no trainable gradients".to_string()))?
            .sqrt()?
            .to_scalar::<f32>()?;
        if !norm.is_finite() {
            candle::bail!("gradient norm is non-finite");
        }
        if f64::from(norm) > max_norm {
            let scale = max_norm / f64::from(norm).max(1e-12);
            for var in &self.vars {
                if let Some(grad) = grads.remove(var.var.as_tensor()) {
                    grads.insert(var.var.as_tensor(), (grad * scale)?);
                }
            }
        }
        self.step(&grads)?;
        Ok(norm)
    }

    pub fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }

    pub fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        let mut tensors = HashMap::new();
        for var in &self.vars {
            tensors.insert(
                state_name("first_moment", &var.name),
                var.first_moment.as_tensor().clone(),
            );
            tensors.insert(
                state_name("second_moment", &var.name),
                var.second_moment.as_tensor().clone(),
            );
        }
        candle::safetensors::save(&tensors, path)
    }

    pub fn load_state(&mut self, path: impl AsRef<Path>, step_t: usize) -> Result<()> {
        let path = path.as_ref();
        let device = self
            .vars
            .first()
            .map(|var| var.var.device().clone())
            .unwrap_or(candle::Device::Cpu);
        let tensors = candle::safetensors::load(path, &device)?;
        for var in &self.vars {
            let first_name = state_name("first_moment", &var.name);
            let second_name = state_name("second_moment", &var.name);
            let first = tensors.get(&first_name).ok_or_else(|| {
                candle::Error::msg(format!(
                    "optimizer state {} missing tensor {first_name}",
                    path.display()
                ))
            })?;
            let second = tensors.get(&second_name).ok_or_else(|| {
                candle::Error::msg(format!(
                    "optimizer state {} missing tensor {second_name}",
                    path.display()
                ))
            })?;
            var.first_moment.set(first)?;
            var.second_moment.set(second)?;
        }
        self.step_t = step_t;
        Ok(())
    }

    pub fn step_t(&self) -> usize {
        self.step_t
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}

fn state_name(kind: &str, var_name: &str) -> String {
    format!("{kind}/{var_name}")
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    use candle::{DType, Device, Result, Tensor};
    use candle_nn::{Init, VarBuilder};

    use super::*;

    #[test]
    fn stateful_adamw_resume_matches_continuous_training() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let params = ParamsAdamW {
            lr: 1e-2,
            weight_decay: 1e-3,
            ..ParamsAdamW::default()
        };
        let continuous = named_scalar_varmap(&device)?;
        let continuous_w = get_weight_tensor(&continuous);
        let mut continuous_opt = StatefulAdamW::new_from_varmap(&continuous, params.clone())?;
        train_simple_step(&continuous_w, &mut continuous_opt)?;
        train_simple_step(&continuous_w, &mut continuous_opt)?;

        let resumed = named_scalar_varmap(&device)?;
        let resumed_w = get_weight_tensor(&resumed);
        let mut resumed_opt = StatefulAdamW::new_from_varmap(&resumed, params)?;
        train_simple_step(&resumed_w, &mut resumed_opt)?;
        let weights_path = temp_path("weights.safetensors");
        let optim_path = temp_path("optim.safetensors");
        resumed.save(&weights_path)?;
        resumed_opt.save_state(&optim_path)?;

        let mut reloaded = named_scalar_varmap(&device)?;
        reloaded.load(&weights_path)?;
        let reloaded_w = get_weight_tensor(&reloaded);
        let mut reloaded_opt =
            StatefulAdamW::new_from_varmap(&reloaded, resumed_opt.params().clone())?;
        reloaded_opt.load_state(&optim_path, resumed_opt.step_t())?;
        train_simple_step(&reloaded_w, &mut reloaded_opt)?;

        let continuous_value = continuous_w.to_vec1::<f32>()?;
        let reloaded_value = reloaded_w.to_vec1::<f32>()?;
        fs::remove_file(weights_path).ok();
        fs::remove_file(optim_path).ok();
        assert_eq!(continuous_opt.step_t(), 2);
        assert_eq!(reloaded_opt.step_t(), 2);
        assert_eq!(continuous_value.len(), reloaded_value.len());
        for (lhs, rhs) in continuous_value.iter().zip(reloaded_value.iter()) {
            assert!((lhs - rhs).abs() < 1e-7, "{lhs} != {rhs}");
        }
        Ok(())
    }

    fn named_scalar_varmap(device: &Device) -> Result<VarMap> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        vb.get_with_hints((2,), "w", Init::Const(1.0))?;
        Ok(varmap)
    }

    fn get_weight_tensor(varmap: &VarMap) -> Tensor {
        varmap
            .data()
            .lock()
            .unwrap()
            .get("w")
            .unwrap()
            .as_tensor()
            .clone()
    }

    fn train_simple_step(weight: &Tensor, opt: &mut StatefulAdamW) -> Result<()> {
        let loss = weight.sqr()?.sum_all()?;
        opt.backward_step(&loss)
    }

    fn temp_path(suffix: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "le-wm-nv-stateful-adamw-{}-{stamp}-{suffix}",
            std::process::id()
        ))
    }
}
