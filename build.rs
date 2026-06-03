use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=NVIDIA_VIDEO_CODEC_SDK_PATH");
    println!("cargo:rerun-if-env-changed=NVIDIA_VIDEO_CODEC_INCLUDE_PATH");

    link_required_nvidia_library("nvjpeg", "libnvjpeg.so");
    link_required_nvidia_library("nvcuvid", "libnvcuvid.so");
}

fn link_required_nvidia_library(link_name: &str, soname: &str) {
    match find_nvidia_library(soname) {
        Some(path) => {
            println!("cargo:rustc-link-search=native={}", path.display());
            println!("cargo:rustc-link-lib={link_name}");
        }
        None => panic!(
            "{soname} is required by le-wm-nv's NVIDIA runtime; set CUDA_HOME, CUDA_PATH, or NVIDIA_VIDEO_CODEC_SDK_PATH"
        ),
    }
}

fn find_nvidia_library(soname: &str) -> Option<PathBuf> {
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Some(root) = env::var_os("CUDA_HOME") {
        roots.push(root.into());
    }
    if let Some(root) = env::var_os("CUDA_PATH") {
        roots.push(root.into());
    }
    if let Some(root) = env::var_os("NVIDIA_VIDEO_CODEC_SDK_PATH") {
        roots.push(root.into());
    }
    if let Some(root) = env::var_os("NVIDIA_VIDEO_CODEC_INCLUDE_PATH") {
        roots.push(root.into());
    }
    roots.push("/usr".into());
    roots.push("/usr/lib".into());
    roots.push("/lib".into());
    roots.push("/usr/local/cuda".into());
    roots.push("/usr/local/cuda-13.0".into());
    roots.push("/usr/local/cuda-12".into());

    for root in roots {
        for rel in [
            "",
            "targets/x86_64-linux/lib",
            "lib64",
            "lib",
            "lib/x86_64-linux-gnu",
            "x86_64-linux-gnu",
        ] {
            let candidate = Path::new(&root).join(rel);
            if candidate.join(soname).exists() {
                return Some(candidate);
            }
        }
    }
    None
}
