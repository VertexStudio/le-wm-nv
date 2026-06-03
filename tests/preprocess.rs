use candle::{DType, Device};
use le_wm_nv::preprocess::{
    ImagePreprocess, RgbFrameShape, preprocess_actions, preprocess_latest_rgb_frame_u8,
    preprocess_rgb_frames_u8,
};

fn device() -> Device {
    Device::new_cuda(0).unwrap()
}

#[test]
fn preprocesses_rgb_frames_to_batched_ncthw_tensor() {
    let frames = [255u8, 0, 127];
    let cfg = ImagePreprocess {
        image_size: 1,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };
    let tensor = preprocess_rgb_frames_u8(
        &frames,
        RgbFrameShape {
            batch: 1,
            time: 1,
            height: 1,
            width: 1,
        },
        cfg,
        DType::F32,
        &device(),
    )
    .unwrap();

    assert_eq!(tensor.shape().dims(), &[1, 1, 3, 1, 1]);
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(values[0], 1.0);
    assert_eq!(values[1], 0.0);
    assert!((values[2] - (127.0 / 255.0)).abs() < 1e-6);
}

#[test]
fn resizes_rgb_frames_with_nearest_neighbor() {
    let frames = [
        10u8, 20, 30, 40, 50, 60, //
        70, 80, 90, 100, 110, 120,
    ];
    let cfg = ImagePreprocess {
        image_size: 1,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };
    let tensor = preprocess_rgb_frames_u8(
        &frames,
        RgbFrameShape {
            batch: 1,
            time: 1,
            height: 2,
            width: 2,
        },
        cfg,
        DType::F32,
        &device(),
    )
    .unwrap();

    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert!((values[0] - (10.0 / 255.0)).abs() < 1e-6);
    assert!((values[1] - (20.0 / 255.0)).abs() < 1e-6);
    assert!((values[2] - (30.0 / 255.0)).abs() < 1e-6);
}

#[test]
fn extracts_latest_rgb_frame_for_pixel_models() {
    let frames = [1u8, 2, 3, 10, 20, 30];
    let cfg = ImagePreprocess {
        image_size: 1,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };
    let tensor = preprocess_latest_rgb_frame_u8(
        &frames,
        RgbFrameShape {
            batch: 1,
            time: 2,
            height: 1,
            width: 1,
        },
        cfg,
        DType::F32,
        &device(),
    )
    .unwrap();

    assert_eq!(tensor.shape().dims(), &[1, 3, 1, 1]);
    let values = tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(values, vec![10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]);
}

#[test]
fn rejects_wrong_rgb_buffer_length() {
    let err = preprocess_rgb_frames_u8(
        &[0u8; 2],
        RgbFrameShape {
            batch: 1,
            time: 1,
            height: 1,
            width: 1,
        },
        ImagePreprocess::imagenet_224(),
        DType::F32,
        &device(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("expected 3"));
}

#[test]
fn clamps_actions_to_bounds() {
    let tensor = preprocess_actions(
        &[-2.0, 0.5, 3.0, 4.0],
        1,
        2,
        2,
        &[-1.0, 0.0],
        &[1.0, 2.0],
        DType::F32,
        &device(),
    )
    .unwrap();

    assert_eq!(tensor.shape().dims(), &[1, 2, 2]);
    assert_eq!(
        tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        vec![-1.0, 0.5, 1.0, 2.0]
    );
}

#[test]
fn rejects_action_bounds_with_wrong_dim() {
    let err = preprocess_actions(
        &[0.0, 1.0],
        1,
        1,
        2,
        &[0.0],
        &[1.0, 1.0],
        DType::F32,
        &device(),
    )
    .unwrap_err();

    assert!(err.to_string().contains("action bounds"));
}
